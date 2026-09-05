# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class PositionProbeCEConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("position_probe_ce", dataclass=PositionProbeCEConfig)
class PositionProbeCECriterion(FairseqCriterion):
    """Cross-entropy criterion for positional probing with direct position indices.

    Target for position t is simply t (0-indexed), no vocabulary mapping.
    The model's output logits should have shape [B, T, T] where class t = position t.
    """

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        logits = net_output[0]  # [B, T, num_positions]
        B, T, C = logits.shape

        # Target = raw position index for each token
        target = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, T)

        lprobs = F.log_softmax(logits.float(), dim=-1)
        loss = F.nll_loss(
            lprobs.view(-1, C),
            target.reshape(-1),
            reduction="sum" if reduce else "none",
        )

        sample_size = B if self.sentence_avg else B * T

        # Accuracy and MAD
        preds = logits.argmax(dim=-1)
        n_correct = (preds == target).sum().item()
        n_total = B * T
        mad_sum = (preds - target).abs().float().sum().item()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": B,
            "sample_size": sample_size,
            "n_correct": n_correct,
            "n_total": n_total,
            "mad_sum": mad_sum,
            "mad_count": n_total,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        n_correct = sum(log.get("n_correct", 0) for log in logging_outputs)
        n_total = sum(log.get("n_total", 0) for log in logging_outputs)
        if n_total > 0:
            metrics.log_scalar(
                "accuracy", 100.0 * n_correct / n_total, n_total, round=2
            )

        mad_sum = sum(log.get("mad_sum", 0) for log in logging_outputs)
        mad_count = sum(log.get("mad_count", 0) for log in logging_outputs)
        if mad_count > 0:
            metrics.log_scalar("mad", mad_sum / mad_count, mad_count, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
