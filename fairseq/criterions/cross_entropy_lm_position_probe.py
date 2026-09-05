# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch as t
from npe_utils import NPE_Utils

@dataclass
class PPCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("dpp_cross_entropy_2", dataclass=PPCrossEntropyCriterionConfig)
class PPCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.positions = NPE_Utils.get_all_positions(task.target_dictionary)  # [i for i in range(8192)]

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        # --- Accuracy and MAD ---
        # compute_loss has set sample['target'] to position class indices
        logits = net_output[0]                     # (B, T, num_classes)
        preds = logits.argmax(dim=-1)              # (B, T)
        target = sample["target"]                  # (B, T)

        n_correct = (preds == target).sum().item()
        n_total = target.numel()

        # Map predicted class indices back to raw positions for MAD
        dim = logits.shape[1]
        num_classes = logits.shape[-1]
        positions_tensor = t.tensor(
            self.positions[:dim], device=logits.device, dtype=t.long
        )
        class_to_pos = t.full(
            (num_classes,), -1, dtype=t.long, device=logits.device
        )
        class_to_pos[positions_tensor] = t.arange(dim, device=logits.device)

        true_pos = t.arange(dim, device=logits.device).unsqueeze(0).expand_as(preds)
        pred_pos = class_to_pos[preds]
        valid = pred_pos >= 0
        mad_sum = (pred_pos[valid] - true_pos[valid]).abs().float().sum().item()
        mad_count = int(valid.sum().item())

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "n_correct": n_correct,
            "n_total": n_total,
            "mad_sum": mad_sum,
            "mad_count": mad_count,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        ##### probe hack #####
        dim = net_output[0].shape[1]
        assert dim <= len(self.positions)

        positions = self.positions[:dim]
        batch_size = net_output[0].shape[0]
        sample['target'] = t.tensor(positions).expand(batch_size, dim).to(sample['target'])
        #######################

        target = model.get_targets(sample, net_output).view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
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

        # Accuracy
        n_correct = sum(log.get("n_correct", 0) for log in logging_outputs)
        n_total = sum(log.get("n_total", 0) for log in logging_outputs)
        if n_total > 0:
            metrics.log_scalar(
                "accuracy", 100.0 * n_correct / n_total, n_total, round=2
            )

        # Mean Absolute Deviation (predicted position vs true position)
        mad_sum = sum(log.get("mad_sum", 0) for log in logging_outputs)
        mad_count = sum(log.get("mad_count", 0) for log in logging_outputs)
        if mad_count > 0:
            metrics.log_scalar("mad", mad_sum / mad_count, mad_count, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
