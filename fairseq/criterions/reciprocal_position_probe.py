# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression probe for position under fixed (Q/K = 0) attention.

With Q/K pinned to zero, causal attention at position t is the prefix mean
``1/(t+1) * sum_{j<=t} v_j``, so ``1/(t+1)`` is the only position-dependent quantity
the network computes. This criterion regresses that quantity, where
``dpp_cross_entropy_fix`` classifies ``t`` and so asks the probe to invert it.

The target is

    y_t = c / (t + a)        t = 0 .. T-1

with two knobs, both of which genuinely change the target -- unlike an additive
``+ b`` outside the fraction, which the probe's own trainable output bias absorbs
exactly (same weights, intercept shifted by b) and which therefore cannot move any
metric.

Scale ``c`` (``--reciprocal-scale``)
    Multiplicative, so it cannot change ``r2``, ``mad`` or ``accuracy`` either -- R^2
    is scale invariant and the inversion divides it out. It exists for conditioning:
    unscaled, late targets are ~2e-3, far from the probe's natural O(1) output range.
    Default (0) resolves to ``c = T / sum_t 1/(t+a)``, making the mean target exactly
    1. The scale is fixed from ``tokens_per_sample``, never a batch's length.

Offset ``a`` (``--reciprocal-offset``, default 1)
    Inside the denominator, so it reshapes the target and is a real knob. It trades
    metric honesty against what a linear probe can express, measured at T=512:

      a       range (max/min)    R^2 of a probe resolving      best linear R^2
                                 only t<10, guessing after     from the 1/(t+1) signal
      1            512x                    0.959                      1.000
      2            257x                    --                         0.947
      10            52x                    0.613                      0.601
      50            11x                    0.277                      0.306

    Larger ``a`` compresses the range and makes R^2 much harder to fool. The cost
    falls on the *linear* arms only: the network computes 1/(t+1), and c/(t+a) is a
    Moebius transform of it, which no linear map can produce, so a perfect linear
    probe is capped at the last column. The MLP arms (``--non-linear-probe``) can
    represent that transform and are not capped.

    ``a = 1`` is the default because it keeps the linear-vs-MLP contrast clean: the
    experiment asks how much of the signal is *linearly* decodable, and at a != 1 a
    linear arm's shortfall mixes that with the target's own nonlinearity. Raising
    ``a`` is reasonable when the MLP arms are what you care about, or to stop R^2
    flattering a probe with no late-position resolution -- just read each linear arm
    against its ceiling rather than against 1.0. ``exp_probe_causal.sh`` prints that
    ceiling for the configured ``a``.

Reading the metrics at a = 1: ``loss`` and ``r2`` are dominated by early positions,
so the prediction is also inverted back to a position,
``t_hat = round(c / pred - a)``, and scored with the same ``mad`` and ``accuracy``
the classification probe logs. Those expose the late-position resolution R^2 hides
and are directly comparable with the classification probe's numbers.

Precision: this probe trains without ``--fp16``. Stored in fp16, perfect predictions
still invert to 100% accuracy up to about a = 200, but the margin shrinks as the
targets bunch together (98% at a = 1000), and fp32 keeps the knob safe across its
whole range on a model this small. The criterion computes in float64 for the same
reason.

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


def default_reciprocal_scale(max_positions, offset=1.0):
    """The scale at which the mean of ``c/(t+a)`` over t < T is exactly 1."""
    denom = sum(1.0 / (t + offset) for t in range(max_positions))
    return max_positions / denom


def reciprocal_probe_stats(pred, valid, scale, offset, max_positions):
    """Loss and summable statistics for a batch.

    Args:
        pred:          ``[B, T]`` float predictions of ``scale / (t + offset)``.
        valid:         ``[B, T]`` bool, False at padded positions.
        scale:         the target scale ``c`` (> 0).
        offset:        the denominator offset ``a`` (> 0).
        max_positions: the configured sequence length; fixes the clamp range.

    Returns:
        ``(loss, stats)`` -- ``loss`` is the summed squared error (fairseq divides by
        the sample size), ``stats`` holds per-batch sums that aggregate exactly across
        batches and workers.
    """
    B, T = pred.shape
    t_idx = torch.arange(T, device=pred.device).unsqueeze(0).expand(B, T)
    target = scale / (t_idx.to(pred.dtype) + offset)

    p = pred[valid]
    r = target[valid]
    ti = t_idx[valid].to(pred.dtype)

    loss = (p - r).pow(2).sum()

    with torch.no_grad():
        # c/(t+a) runs from c/a at t=0 down to c/(T-1+a) at the last position. Clamp
        # into that range before inverting, so a prediction at or below zero maps to
        # the last position rather than to inf or a negative index.
        clamped = p.clamp(min=scale / (max_positions - 1 + offset), max=scale / offset)
        t_hat = torch.round(scale / clamped - offset)
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
            "help": "scale c in the target c/(t+a). 0 (default) resolves to "
            "T / sum_t 1/(t+a), which makes the mean target exactly 1. Must be >= 0. "
            "Changes only the loss's units and the optimisation, never r2/mad/accuracy."
        },
    )
    reciprocal_offset: float = field(
        default=1.0,
        metadata={
            "help": "offset a in the denominator of c/(t+a). Must be > 0. a=1 is the "
            "prefix-mean weight the network actually computes, and the only value a "
            "linear probe can fit exactly; larger a compresses the target's range and "
            "makes r2 harder to fool, but caps a perfect linear probe (R^2 <= 0.60 at "
            "a=10, 0.31 at a=50), so raise it only with --non-linear-probe."
        },
    )
    tokens_per_sample: int = II("task.tokens_per_sample")


@register_criterion("reciprocal_position_probe", dataclass=ReciprocalPositionProbeConfig)
class ReciprocalPositionProbeCriterion(FairseqCriterion):
    def __init__(
        self, task, reciprocal_scale=0.0, reciprocal_offset=1.0, tokens_per_sample=512
    ):
        super().__init__(task)
        self.max_positions = int(tokens_per_sample)

        # a <= 0 divides by zero at t = -a, which is inside the sequence.
        assert reciprocal_offset > 0, (
            f"--reciprocal-offset must be > 0, got {reciprocal_offset}"
        )
        # A negative scale reverses the ordering the inversion relies on.
        assert reciprocal_scale >= 0, (
            f"--reciprocal-scale must be >= 0 (0 = auto), got {reciprocal_scale}"
        )
        self.offset = float(reciprocal_offset)
        self.scale = (
            float(reciprocal_scale)
            if reciprocal_scale > 0
            else default_reciprocal_scale(self.max_positions, self.offset)
        )

        logger.info(
            "reciprocal probe: target = %.4f / (t + %.4f) over T=%d (range %.4f .. %.4f)",
            self.scale,
            self.offset,
            self.max_positions,
            self.scale / (self.max_positions - 1 + self.offset),
            self.scale / self.offset,
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
        # float64 for the criterion's own arithmetic: neighbouring late positions
        # differ by ~c/T^2 in the target, and the inversion divides by the prediction.
        # The loss goes back as float32.
        pred = pred.squeeze(-1).double()
        valid = sample["target"].ne(self.padding_idx)

        loss, stats = reciprocal_probe_stats(
            pred, valid, self.scale, self.offset, self.max_positions
        )
        loss = loss.float()
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
