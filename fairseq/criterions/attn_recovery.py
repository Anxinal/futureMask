# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression loss for the self-attention recovery experiment.

Trains a masked student (``simple_attn_student``) to reproduce a frozen unmasked
teacher's attention sublayer output. The loss is L1, as the experiment plan
specifies; the interesting numbers are the metrics logged beside it.

Metrics, all computed over non-padded positions:

  loss / mae        mean absolute error per dimension.
  rel_l2            ||student - teacher|| / ||teacher||. The headline number: it is
                    scale-free, so it stays comparable across conditions and across
                    teachers whose activations have different magnitudes.
  cos               mean cosine similarity between the two vectors.
  mae_baseline      error of the best *constant* prediction (the batch-mean teacher
                    vector). Without it an mae of 0.02 means nothing -- this says how
                    much of the fit is real structure rather than a well-placed
                    constant.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class AttnRecoveryCriterionConfig(FairseqDataclass):
    pass


@register_criterion("attn_recovery", dataclass=AttnRecoveryCriterionConfig)
class AttnRecoveryCriterion(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        student = net_output[0]
        extra = net_output[1]
        assert "teacher_attn_out" in extra, (
            "attn_recovery expects a model whose forward returns the frozen teacher's "
            "attention output, e.g. --arch simple_attn_student_base"
        )
        teacher = extra["teacher_attn_out"]
        padding_mask = extra.get("padding_mask", None)

        if padding_mask is not None:
            valid = ~padding_mask
            student = student[valid]
            teacher = teacher[valid]
        else:
            student = student.reshape(-1, student.size(-1))
            teacher = teacher.reshape(-1, teacher.size(-1))

        # Metrics in fp32: under --fp16 the squared sums below overflow easily.
        student = student.float()
        teacher = teacher.float()
        diff = student - teacher

        # Mean over dimensions, summed over tokens, so the logged loss reads as the
        # per-dimension MAE once fairseq divides by sample_size.
        loss = diff.abs().mean(dim=-1).sum()
        n_valid = student.size(0)

        with torch.no_grad():
            sq_err = diff.pow(2).sum()
            sq_ref = teacher.pow(2).sum()
            cos_sum = F.cosine_similarity(student, teacher, dim=-1).sum()
            baseline_sum = (
                (teacher - teacher.mean(dim=0, keepdim=True)).abs().mean(dim=-1).sum()
            )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": n_valid,
            "sq_err": sq_err.data,
            "sq_ref": sq_ref.data,
            "cos_sum": cos_sum.data,
            "baseline_sum": baseline_sum.data,
        }
        return loss, n_valid, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sq_err = sum(log.get("sq_err", 0) for log in logging_outputs)
        sq_ref = sum(log.get("sq_ref", 0) for log in logging_outputs)
        cos_sum = sum(log.get("cos_sum", 0) for log in logging_outputs)
        baseline_sum = sum(log.get("baseline_sum", 0) for log in logging_outputs)

        if sample_size == 0:
            return

        # No log(2) conversion here: this is an L1 distance, not a log probability.
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("mae", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar(
            "mae_baseline", baseline_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar("cos", cos_sum / sample_size, sample_size, round=5)
        if sq_ref > 0:
            metrics.log_scalar(
                "rel_l2", math.sqrt(sq_err / sq_ref), sample_size, round=5
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
