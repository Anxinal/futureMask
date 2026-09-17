# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression probe for position under fixed (Q/K = 0) attention.

With Q/K pinned to zero, causal attention at position t is the prefix mean
``1/(t+1) * sum_{j<=t} v_j``, so ``1/(t+1)`` is the only position-dependent quantity
the network computes. This criterion regresses exactly that quantity, where
``dpp_cross_entropy_fix`` classifies ``t`` and so asks the probe to invert it.

The target is scaled: ``c / (t+1)``. Unscaled, late targets are ~2e-3 and their squared
errors ~1e-6, and since Adam moves each weight by roughly ``lr`` per step whatever the
gradient's size, a fixed update budget leaves the probe fitting targets far from its
natural O(1) output range. By default ``c = T / H_T`` (``H_T`` the T-th harmonic number),
which makes the mean target over positions exactly 1 -- about 75 down to 0.15 at T=512.
The scale is fixed from ``tokens_per_sample``, never from a batch's length, so it cannot
drift between batches.

A constant scale leaves ``r2``, ``mad`` and ``accuracy`` mathematically unchanged; only
``loss`` (the MSE, in scaled units) depends on it.

Two things to know when reading the metrics:

* ``loss`` and ``r2`` are dominated by early positions. The target falls into a thin
  sliver by t = 50 and every later position sits inside it, so a probe that resolves
  only the first 10 positions and guesses a constant after still scores R^2 ~ 0.96
  while being off by ~157 positions on average.
* So the prediction is also inverted back to a position, ``t_hat = round(c/pred) - 1``,
  and scored with the same ``mad`` and ``accuracy`` the classification probe logs. Most
  positions are late, so these expose exactly the resolution R^2 hides -- and they are
  directly comparable with the classification probe's numbers.

Use with ``--arch fixed_attn_probe --probe-target reciprocal``, which gives the probe a
single output unit.
"""

import logging
from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)


def default_reciprocal_scale(max_positions):
    """``T / H_T``: the scale at which the mean of ``c/(t+1)`` over t < T is exactly 1."""
    harmonic = sum(1.0 / i for i in range(1, max_positions + 1))
    return max_positions / harmonic


def reciprocal_probe_stats(pred, valid, scale, max_positions):
    """Loss and summable statistics for a batch.

    Args:
        pred:          ``[B, T]`` float predictions of ``scale / (t+1)``.
        valid:         ``[B, T]`` bool, False at padded positions.
        scale:         the target scale ``c``.
        max_positions: the configured sequence length; fixes the clamp range.

    Returns:
        ``(loss, stats)`` -- ``loss`` is the summed squared error (fairseq divides by
        the sample size), ``stats`` holds per-batch sums that aggregate exactly across
        batches and workers.
    """
    B, T = pred.shape
    t_idx = torch.arange(T, device=pred.device).unsqueeze(0).expand(B, T)
    target = scale / (t_idx.to(pred.dtype) + 1.0)

    p = pred[valid]
    r = target[valid]
    ti = t_idx[valid].to(pred.dtype)

    loss = (p - r).pow(2).sum()

    with torch.no_grad():
        # c/(t+1) lives in [c/T, c]. Clamp before inverting, so a prediction at or
        # below zero maps to the last position rather than to inf or a negative index.
        clamped = p.clamp(min=scale / max_positions, max=scale)
        t_hat = torch.round(scale / clamped) - 1
        stats = {
            "rp_total": p.numel(),
            "rp_sq_err": loss.detach(),
            "rp_r_sum": r.sum(),
            "rp_r_sq_sum": r.pow(2).sum(),
            "rp_abs_diff": (t_hat - ti).abs().sum(),
            "rp_correct": (t_hat == ti).sum(),
        }
    return loss, stats


@dataclass
class ReciprocalPositionProbeConfig(FairseqDataclass):
    reciprocal_scale: float = field(
        default=0.0,
        metadata={
            "help": "multiply the 1/(t+1) target by this constant. 0 (default) uses "
            "T/H_T, which makes the mean target exactly 1. Changes only the loss's "
            "units and the optimisation, never r2, mad or accuracy."
        },
    )
    tokens_per_sample: int = II("task.tokens_per_sample")


@register_criterion("reciprocal_position_probe", dataclass=ReciprocalPositionProbeConfig)
class ReciprocalPositionProbeCriterion(FairseqCriterion):
    def __init__(self, task, reciprocal_scale=0.0, tokens_per_sample=512):
        super().__init__(task)
        self.max_positions = int(tokens_per_sample)
        self.scale = (
            float(reciprocal_scale)
            if reciprocal_scale and reciprocal_scale > 0
            else default_reciprocal_scale(self.max_positions)
        )
        logger.info(
            "reciprocal probe: target = %.4f / (t+1) over T=%d (range %.4f .. %.4f)",
            self.scale,
            self.max_positions,
            self.scale / self.max_positions,
            self.scale,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        pred = net_output[0]
        assert pred.size(-1) == 1, (
            f"reciprocal_position_probe expects a 1-unit probe head, got "
            f"{pred.size(-1)} outputs -- pass --probe-target reciprocal"
        )
        pred = pred.squeeze(-1).float()
        valid = sample["target"].ne(self.padding_idx)

        loss, stats = reciprocal_probe_stats(
            pred, valid, self.scale, self.max_positions
        )
        sample_size = int(stats["rp_total"])

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        for k, v in stats.items():
            logging_output[k] = utils.item(v) if torch.is_tensor(v) else v
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        def total(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        n = total("rp_total")
        if n == 0:
            return

        # Plain per-token MSE in scaled units -- no log(2) conversion, this is not a
        # log probability.
        metrics.log_scalar("loss", total("rp_sq_err") / n, n, round=6)

        # Accumulate raw sums and derive the ratios from them (the pattern
        # dpp_cross_entropy_fix uses for accuracy), so r2, mad and accuracy are exact
        # over the whole validation set rather than averages of per-batch values --
        # which for R^2 would not be the same number.
        for key in ("rp_total", "rp_sq_err", "rp_r_sum", "rp_r_sq_sum",
                    "rp_abs_diff", "rp_correct"):
            metrics.log_scalar(key, total(key), round=6)

        def r2(meters):
            count = meters["rp_total"].sum
            sst = meters["rp_r_sq_sum"].sum - meters["rp_r_sum"].sum ** 2 / count
            if sst <= 0:
                return float("nan")
            return round(1.0 - meters["rp_sq_err"].sum / sst, 5)

        metrics.log_derived("r2", r2)
        metrics.log_derived(
            "mad",
            lambda meters: round(meters["rp_abs_diff"].sum / meters["rp_total"].sum, 3),
        )
        metrics.log_derived(
            "accuracy",
            lambda meters: round(
                100.0 * meters["rp_correct"].sum / meters["rp_total"].sum, 3
            ),
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
