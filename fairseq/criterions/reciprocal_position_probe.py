# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression probe for position under fixed (Q/K = 0) attention.

With Q/K pinned to zero, causal attention at position t is the prefix mean
``1/(t+1) * sum_{j<=t} v_j``, so ``1/(t+1)`` is the only position-dependent quantity
the network computes. This criterion regresses that quantity, where
``dpp_cross_entropy_fix`` classifies ``t`` and so asks the probe to invert it.

Target
------

    y_t = c / (t + a)        t = 0 .. T-1

``c`` (``--reciprocal-scale``, 0 = auto = ``T / sum_t 1/(t+a)``, which puts the mean
target at 1) is multiplicative and cannot change r2/mad/accuracy. ``a``
(``--reciprocal-offset``, default 1) sits inside the denominator and does reshape the
target: larger ``a`` compresses its range (512x at a=1, 52x at a=10, 11x at a=50) and
stops R^2 flattering a probe with no late-position resolution, at the price of capping
the *linear* arms -- c/(t+a) is a Moebius transform of the 1/(t+1) the network
computes, which no linear map can produce (ceiling R^2 0.60 at a=10, 0.31 at a=50).
The ``*_mlp`` arms are uncapped. ``exp_probe_causal.sh`` prints that ceiling.

Loss weighting
--------------

Plain MSE on this target is almost blind to late positions, which is what made an
early run post loss 0.15 / R^2 0.96 while its MAD was still 70 positions out of 512:

* the target spans 103x from t=0 to t=511 (at a=5), so **75% of its variance lives in
  the first 10 positions** and 92% in the first 50;
* the inversion amplifies error quadratically, ``|dt/dy| = (t+a)^2/c``, so 0.01 of
  target error is 0.3 positions at t=50 but 24.7 positions at t=511;
* being 70 positions wrong at t=300 is a target error of 0.066, which is 0.01% of the
  total loss budget. The optimizer has no reason to fix it, and does not.

``--reciprocal-loss-weight jacobian`` (the default) fixes that by weighting each
token's squared error with the squared Jacobian ``((t+a)^2/c)^2``, so the loss becomes
the squared error *in position space*: every position then contributes on the same
scale, whatever its target magnitude. The weights are normalised to mean 1 over the
configured T, which keeps the loss and its gradients in the same numeric range as the
unweighted version -- a constant factor that changes neither the optimum nor the
relative weighting between positions. ``none`` restores plain MSE.

Expect the trade this buys: R^2 and the raw MSE are target-space quantities dominated
by early positions, so **they can get worse while mad, accuracy and pos_rmse improve**.
That is the intended direction, not a regression.

Metrics
-------

``loss``      weighted (or plain) MSE in target units.
``r2``        target-space explained variance, unweighted, so it stays comparable
              across weightings. Read it together with the position-space numbers.
``pos_rmse``  RMS of the first-order position error ``((t+a)^2/c) * (pred - y)``.
              Directly comparable with ``mad``; the gap between them shows how much
              of the error sits in a few bad positions.
``mad`` /     exact inversion ``t_hat = round(c/pred - a)`` scored against t, the same
``accuracy``  metrics the classification probe logs.
``mad_q1..4`` ``mad`` within each quarter of the sequence -- where the error actually
              is. A single ``mad`` hides a strong position gradient.

Precision: this probe trains without ``--fp16`` (see ``exp_probe_causal.sh``), and the
criterion computes in float64.

Use with ``--arch fixed_attn_probe --probe-target reciprocal``.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)

N_BUCKETS = 4


def default_reciprocal_scale(max_positions, offset=1.0):
    """The scale at which the mean of ``c/(t+a)`` over t < T is exactly 1."""
    denom = sum(1.0 / (t + offset) for t in range(max_positions))
    return max_positions / denom


@lru_cache(maxsize=16)
def _position_weights(max_positions, scale, offset):
    """``[T]`` squared-Jacobian weights, normalised to mean 1, on CPU (float64).

    ``|dt/dy| = (t+a)^2/c``, so weighting a squared target error by its square turns
    the loss into squared position error. Normalising by the mean over the configured
    T keeps the loss's scale comparable to the unweighted version and constant across
    batches -- a short final batch must not change the loss's units.
    """
    t = torch.arange(max_positions, dtype=torch.float64)
    w = ((t + offset) ** 2 / scale) ** 2
    return w / w.mean()


@lru_cache(maxsize=16)
def _jacobian(max_positions, scale, offset):
    """``[T]`` un-normalised ``|dt/dy|``, for reporting error in positions."""
    t = torch.arange(max_positions, dtype=torch.float64)
    return (t + offset) ** 2 / scale


def reciprocal_probe_stats(pred, valid, scale, offset, max_positions, weighted=True):
    """Loss and summable statistics for a batch.

    Args:
        pred:          ``[B, T]`` float predictions of ``scale / (t + offset)``.
        valid:         ``[B, T]`` bool, False at padded positions.
        scale:         the target scale ``c`` (> 0).
        offset:        the denominator offset ``a`` (> 0).
        max_positions: the configured sequence length; fixes the weights and clamps.
        weighted:      apply the squared-Jacobian weighting.

    Returns:
        ``(loss, stats)`` -- ``loss`` is the summed (weighted) squared error, which
        fairseq divides by the sample size; ``stats`` holds per-batch sums that
        aggregate exactly across batches and workers.
    """
    B, T = pred.shape
    t_idx = torch.arange(T, device=pred.device).unsqueeze(0).expand(B, T)
    target = scale / (t_idx.to(pred.dtype) + offset)

    p = pred[valid]
    r = target[valid]
    ti_long = t_idx[valid]
    ti = ti_long.to(pred.dtype)

    err_sq = (p - r).pow(2)
    if weighted:
        w = _position_weights(max_positions, scale, offset).to(
            device=pred.device, dtype=pred.dtype
        )[ti_long]
        loss = (w * err_sq).sum()
    else:
        loss = err_sq.sum()

    with torch.no_grad():
        # First-order error in positions: |dt/dy| * |dy|. Reported as pos_rmse.
        jac = _jacobian(max_positions, scale, offset).to(
            device=pred.device, dtype=pred.dtype
        )[ti_long]
        pos_sq = (jac * (p - r)).pow(2)

        # c/(t+a) runs from c/a at t=0 down to c/(T-1+a). Clamp into that range before
        # inverting, so a prediction at or below zero maps to the last position rather
        # than to inf or a negative index.
        clamped = p.clamp(min=scale / (max_positions - 1 + offset), max=scale / offset)
        t_hat = torch.round(scale / clamped - offset)
        abs_diff = (t_hat - ti).abs()

        stats = {
            "rp_total": p.numel(),
            "rp_sq_err": err_sq.sum(),
            "rp_weighted_err": loss.detach(),
            "rp_pos_sq_err": pos_sq.sum(),
            "rp_r_sum": r.sum(),
            "rp_r_sq_sum": r.pow(2).sum(),
            "rp_abs_diff": abs_diff.sum(),
            "rp_correct": (t_hat == ti).sum(),
        }

        # Where the error actually is: mad within each quarter of the sequence.
        bucket = (ti_long * N_BUCKETS // max_positions).clamp(max=N_BUCKETS - 1)
        for b in range(N_BUCKETS):
            sel = bucket == b
            stats[f"rp_mad_q{b + 1}"] = abs_diff[sel].sum()
            stats[f"rp_n_q{b + 1}"] = sel.sum()

    return loss, stats


@dataclass
class ReciprocalPositionProbeConfig(FairseqDataclass):
    reciprocal_scale: float = field(
        default=0.0,
        metadata={
            "help": "scale c in the target c/(t+a). 0 (default) resolves to "
            "T / sum_t 1/(t+a), which makes the mean target exactly 1. Must be >= 0."
        },
    )
    reciprocal_offset: float = field(
        default=1.0,
        metadata={
            "help": "offset a in the denominator of c/(t+a). Must be > 0. a=1 is the "
            "prefix-mean weight the network computes and the only value a linear probe "
            "can fit exactly; larger a compresses the target's range but caps the "
            "linear arms (R^2 <= 0.60 at a=10, 0.31 at a=50)."
        },
    )
    reciprocal_loss_weight: str = field(
        default="jacobian",
        metadata={
            "help": "'jacobian' (default) weights each squared error by ((t+a)^2/c)^2, "
            "so the loss is squared error in POSITION space and every position "
            "contributes on the same scale; plain MSE is near-blind to late positions "
            "(0.01 of target error is 0.3 positions at t=50 but 24.7 at t=511). "
            "'none' restores plain MSE. Expect r2 (target space) to fall while mad and "
            "pos_rmse improve."
        },
    )
    tokens_per_sample: int = II("task.tokens_per_sample")


@register_criterion("reciprocal_position_probe", dataclass=ReciprocalPositionProbeConfig)
class ReciprocalPositionProbeCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        reciprocal_scale=0.0,
        reciprocal_offset=1.0,
        reciprocal_loss_weight="jacobian",
        tokens_per_sample=512,
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
        assert reciprocal_loss_weight in ("jacobian", "none"), (
            f"--reciprocal-loss-weight must be 'jacobian' or 'none', got "
            f"{reciprocal_loss_weight!r}"
        )
        self.offset = float(reciprocal_offset)
        self.scale = (
            float(reciprocal_scale)
            if reciprocal_scale > 0
            else default_reciprocal_scale(self.max_positions, self.offset)
        )
        self.weighted = reciprocal_loss_weight == "jacobian"

        logger.info(
            "reciprocal probe: target = %.4f / (t + %.4f) over T=%d (range %.4f .. %.4f), "
            "loss weighting = %s",
            self.scale,
            self.offset,
            self.max_positions,
            self.scale / (self.max_positions - 1 + self.offset),
            self.scale / self.offset,
            reciprocal_loss_weight,
        )
        if self.weighted:
            w = _position_weights(self.max_positions, self.scale, self.offset)
            logger.info(
                "  squared-Jacobian weights (mean 1): %.2e at t=0, %.4f at t=T/2, "
                "%.2f at t=T-1",
                w[0].item(),
                w[self.max_positions // 2].item(),
                w[-1].item(),
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
            pred, valid, self.scale, self.offset, self.max_positions, self.weighted
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

        # The optimised quantity, per token. No log(2) conversion -- not a log prob.
        metrics.log_scalar("loss", total("rp_weighted_err") / n, n, round=6)

        # Accumulate raw sums and derive the ratios from them (the pattern
        # dpp_cross_entropy_fix uses for accuracy), so r2, mad, pos_rmse and the
        # per-quarter mads are exact over the whole validation set rather than
        # averages of per-batch values -- which for R^2 would not be the same number.
        keys = ["rp_total", "rp_sq_err", "rp_weighted_err", "rp_pos_sq_err",
                "rp_r_sum", "rp_r_sq_sum", "rp_abs_diff", "rp_correct"]
        keys += [f"rp_mad_q{b + 1}" for b in range(N_BUCKETS)]
        keys += [f"rp_n_q{b + 1}" for b in range(N_BUCKETS)]
        for key in keys:
            metrics.log_scalar(key, total(key), round=6)

        def r2(meters):
            # Unweighted, target space, so it stays comparable across loss weightings.
            count = meters["rp_total"].sum
            sst = meters["rp_r_sq_sum"].sum - meters["rp_r_sum"].sum ** 2 / count
            if sst <= 0:
                return float("nan")
            return round(1.0 - meters["rp_sq_err"].sum / sst, 5)

        metrics.log_derived("r2", r2)
        metrics.log_derived(
            "mse",
            lambda meters: round(meters["rp_sq_err"].sum / meters["rp_total"].sum, 6),
        )
        metrics.log_derived(
            "pos_rmse",
            lambda meters: round(
                (meters["rp_pos_sq_err"].sum / meters["rp_total"].sum) ** 0.5, 3
            ),
        )
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
        for b in range(N_BUCKETS):
            def _mad_q(meters, b=b):
                cnt = meters[f"rp_n_q{b + 1}"].sum
                if cnt <= 0:
                    return float("nan")
                return round(meters[f"rp_mad_q{b + 1}"].sum / cnt, 3)

            metrics.log_derived(f"mad_q{b + 1}", _mad_q)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
