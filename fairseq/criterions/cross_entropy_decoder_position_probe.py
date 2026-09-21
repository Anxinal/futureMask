# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import random

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch as t
from npe_utils import NPE_Utils
from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class DPPCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    dont_report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )

    distance_penalty: float = field(
        default=0.0,
        metadata={
            "help": "lambda in  loss = CE + lambda * E_p[|k - t|] / T. Cross-entropy is "
            "blind to ordering -- predicting 301 for 300 costs the same as predicting "
            "0 -- so this adds the expected position distance under the predicted "
            "distribution. Its gradient w.r.t. logit j is p_j (d_j - E_p[d]): every "
            "class is pushed down in proportion to how much farther it is than the "
            "current expected distance. Absolute distance, so every position is "
            "weighted alike. 0 (default) is plain cross-entropy."
        },
    )


@register_criterion("dpp_cross_entropy_fix", dataclass=DPPCrossEntropyCriterionConfig)
class DPPCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, dont_report_accuracy=False, distance_penalty=0.0):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.positions = [i for i in range(8192)]  # We start from 5 to ignore the special tokens
        self.report_accuracy = not dont_report_accuracy
        assert distance_penalty >= 0, (
            f"--distance-penalty must be >= 0, got {distance_penalty}"
        )
        self.distance_penalty = float(distance_penalty)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        #if False:
        # all tokens are the same probe
        #input_number = random.randint(0,sample["net_input"]['src_tokens'].max().item())
        #print("input:", input_number)
        #sample["net_input"]['src_tokens'] = t.zeros_like(sample["net_input"]['src_tokens']) + input_number
        if False:
            hack = t.zeros_like(sample["net_input"]['src_tokens'])
            i = 0
            while 2*i+1 < len(sample["net_input"]['src_tokens'][0]):
                hack[0][2*i] = sample["net_input"]['src_tokens'][0][i]
                hack[0][2*i + 1] = sample["net_input"]['src_tokens'][0][i]
                i += 1
            sample["net_input"]['src_tokens'] = hack

        net_output = model(**sample["net_input"])

        lprobs, target = get_lprobs_and_target(self.positions, self.padding_idx, model, net_output, sample)

        nll, _ = self.compute_loss(lprobs, target, reduce=reduce)
        ordinal = ordinal_stats(
            lprobs, target, self.padding_idx, model.decoder.dictionary.nspecial + 1
        )

        # The distance penalty is a loss term, so autograd delivers the
        # distance-proportional gradient -- no custom backward needed.
        if self.distance_penalty > 0:
            if reduce:
                loss = nll + self.distance_penalty * ordinal["exp_dist"].sum()
            else:
                per_token = t.zeros_like(nll)
                per_token[target.ne(self.padding_idx)] = ordinal["exp_dist"]
                loss = nll + self.distance_penalty * per_token
        else:
            loss = nll

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ce_sum": nll.sum().data,
            "exp_dist_sum": utils.item(ordinal["exp_dist"].sum().data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(lprobs, target)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

            abs_diff_sum = self.compute_mean_absolute_difference(lprobs, target)
            logging_output["abs_diff_sum"] = utils.item(abs_diff_sum.data)

            logging_output["abs_diff_median_sum"] = utils.item(ordinal["abs_median"].sum().data)
            for b in range(N_BUCKETS):
                sel = ordinal["bucket"] == b
                logging_output[f"mad_q{b + 1}_sum"] = utils.item(ordinal["abs_argmax"][sel].sum().data)
                logging_output[f"n_q{b + 1}"] = utils.item(sel.sum().data)

        # Print out some examples:
        #lprobs, target = get_lprobs_and_target(self.positions, self.padding_idx, model, net_output, sample)
        #outputs = [[j.item()-5 for j in [i.data for i in lprobs.view(net_output[0].shape).argmax(2)][k]] for k in range(net_output[0].shape[0])]
        #for output in outputs:
        #    print("O: ", output)
        return loss, sample_size, logging_output

    def compute_loss(self, lprobs, target, reduce=True):
        # Catch a probe head that is too small for the position labels here, with a
        # readable message, rather than as an async CUDA device-side assert later.
        assert int(target.max()) < lprobs.size(-1), (
            f"position target {int(target.max())} is out of range for a probe head "
            f"with {lprobs.size(-1)} classes"
        )
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    def compute_accuracy(self, lprobs, target):
        mask = target.ne(self.padding_idx)
        n_correct = t.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        #print(lprobs.argmax(1).masked_select(mask))
        total = t.sum(mask)
        return n_correct, total

    def two_bos_hack(sample):
        hack = t.zeros_like(sample["net_input"]['src_tokens'])
        hack[0][0] = sample["net_input"]['src_tokens'][0][0]
        hack[0][1] = sample["net_input"]['src_tokens'][0][0]
        hack[0][2] = sample["net_input"]['src_tokens'][0][0]
        hack[0][3:] = sample["net_input"]['src_tokens'][0][2:-1]

        return hack

    def all_double_bos_hack(sample):
        hack = t.zeros_like(sample["net_input"]['src_tokens'])
        for i in range(round(len(sample["net_input"]['src_tokens'][0])/2)):
            hack[0][i] = sample["net_input"]['src_tokens'][0][i]
            hack[0][i+1] = sample["net_input"]['src_tokens'][0][i]
        return hack

    def compute_mean_absolute_difference(self, lprobs, target):
        mask = target.ne(self.padding_idx)
        abs_diff_sum = (target.masked_select(mask) - lprobs.argmax(1).masked_select(mask)).abs().sum()
        return abs_diff_sum


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

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            abs_diff_sum = utils.item(
                sum(log.get("abs_diff_sum", 0) for log in logging_outputs)
            )
            metrics.log_scalar("abs_diff_sum", abs_diff_sum)
            metrics.log_derived(
                "mad",
                lambda meters: round(
                    meters["abs_diff_sum"].sum / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

            # Median of the predicted distribution: the readout that minimises
            # expected absolute error, so the natural companion to mad. When the probe
            # spreads mass over neighbouring late positions, the argmax picks one of
            # them somewhat arbitrarily; the median does not.
            metrics.log_scalar(
                "abs_diff_median_sum",
                utils.item(sum(log.get("abs_diff_median_sum", 0) for log in logging_outputs)),
            )
            metrics.log_derived(
                "mad_median",
                lambda meters: round(
                    meters["abs_diff_median_sum"].sum / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

            # mad (argmax) within each quarter of the sequence -- where the error is.
            for b in range(N_BUCKETS):
                metrics.log_scalar(
                    f"mad_q{b + 1}_sum",
                    utils.item(sum(log.get(f"mad_q{b + 1}_sum", 0) for log in logging_outputs)),
                )
                metrics.log_scalar(
                    f"n_q{b + 1}",
                    utils.item(sum(log.get(f"n_q{b + 1}", 0) for log in logging_outputs)),
                )

                def _mad_q(meters, b=b):
                    cnt = meters[f"n_q{b + 1}"].sum
                    return round(meters[f"mad_q{b + 1}_sum"].sum / cnt, 3) if cnt > 0 else float("nan")

                metrics.log_derived(f"mad_q{b + 1}", _mad_q)

        # Pure cross-entropy in bits, and the expected normalised distance, logged
        # separately so both stay readable when distance_penalty > 0 -- "loss" is then
        # their weighted sum. With distance_penalty = 0, ce equals loss.
        ce_sum = sum(log.get("ce_sum", 0) for log in logging_outputs)
        metrics.log_scalar("ce", ce_sum / sample_size / math.log(2), sample_size, round=3)
        if total > 0:
            exp_dist_sum = sum(log.get("exp_dist_sum", 0) for log in logging_outputs)
            metrics.log_scalar("exp_dist", exp_dist_sum / total, total, round=5)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


N_BUCKETS = 4


def ordinal_stats(lprobs, target, padding_idx, class_offset):
    """Position-aware quantities over the non-padded tokens.

    Class index k encodes position k - class_offset (``get_lprobs_and_target`` shifts
    positions past the special symbols). Returns, per valid token:

      exp_dist    E_p[|k - t|] / T, the expected normalised distance -- the penalty
                  term, differentiable through the softmax.
      abs_argmax  |argmax - t| in positions (no grad).
      abs_median  |median of p - t| in positions (no grad).
      bucket      which quarter of the sequence t falls in (no grad).
    """
    valid = target.ne(padding_idx)
    lp = lprobs[valid]                                   # [N, C], float32
    tgt = target[valid]                                  # [N]
    n_classes = lp.size(-1)
    n_pos = n_classes - class_offset                     # positions the head expresses

    p = lp.exp()
    class_pos = t.arange(n_classes, device=lp.device, dtype=lp.dtype) - class_offset
    true_pos = (tgt - class_offset).to(lp.dtype)
    dist = (class_pos.unsqueeze(0) - true_pos.unsqueeze(1)).abs() / n_pos
    exp_dist = (p * dist).sum(-1)

    with t.no_grad():
        argmax_pos = (lp.argmax(-1) - class_offset).to(lp.dtype)
        # First class at which the CDF reaches 1/2. argmax over a bool tensor returns
        # the first True.
        median_pos = ((p.cumsum(-1) >= 0.5).to(lp.dtype).argmax(-1) - class_offset).to(lp.dtype)
        bucket = (true_pos.long() * N_BUCKETS // n_pos).clamp(0, N_BUCKETS - 1)

    return {
        "exp_dist": exp_dist,
        "abs_argmax": (argmax_pos - true_pos).abs(),
        "abs_median": (median_pos - true_pos).abs(),
        "bucket": bucket,
    }


def get_lprobs_and_target(all_positions, padding_idx, model, net_output, sample):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)


    ##### probe hack #####
    batch_size, dim = net_output[0].shape[0:2]
    assert dim <= len(all_positions)
    positions = all_positions[:dim]
    mask = sample['target'].eq(padding_idx)

    # this can only happen in debug!
    #if mask.shape[0] != batch_size:
    #    mask = mask[0].expand(batch_size,dim)

    # Shift positions past the special symbols FIRST, then mark padding. The other
    # order (fill with padding_idx, then add the offset) turned every padded slot into
    # class padding_idx + nspecial + 1, so ignore_index no longer skipped it: padded
    # tokens were trained to predict position 1 and counted in accuracy and mad.
    # Class indices start at nspecial + 1 > padding_idx, so a real position can never
    # collide with it.
    pos_target = t.tensor(positions).expand(batch_size, dim).to(sample['target'])
    pos_target = pos_target + model.decoder.dictionary.nspecial + 1
    sample['target'] = pos_target.masked_fill(mask, padding_idx)
    #######################

    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = model.get_targets(sample, net_output).view(-1)
    return lprobs, target
