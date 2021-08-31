# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion("label_smoothed_cross_entropy_with_denoising")
class LabelSmoothedCrossEntropyCriterionWithDenoising(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 denoising_loss_ratio=0.0,
                 denoising_loss_type=None):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.denoising_loss_ratio = denoising_loss_ratio
        self.denoising_loss_type = denoising_loss_type
        if self.denoising_loss_type == 'round_by_step':
            self.round = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--denoising-loss-ratio",
            default=1.0,
            type=float,
            metavar="D",
            help="weight for the denoising loss",
        )
        parser.add_argument(
            "--denoising-loss-type",
            default="",
            type=str,
            help="type of calculate denoising loss, (round or simultaneous)"
        )

    def forward(self, model, sample, reduce=True, task_type='translation_and_denoising', epoch=None):

        """Compute the loss for the given sample.
        task_type: [only_translation, only_denoising, translation_and_denoising]

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        loss = torch.zeros(1)
        nll_loss = torch.zeros(1)
        translation_loss = torch.zeros(1)
        translation_nll_loss = torch.zeros(1)
        denoising_loss = torch.zeros(1)
        denoising_nll_loss = torch.zeros(1)

        if task_type == 'only_denoising':
            net_input = {"src_tokens": sample["net_input"]['noisy_target'],
                         "src_lengths": sample["net_input"]['noisy_target_lengths'],
                         "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
            net_output = model(**net_input)
            denoising_loss, denoising_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = denoising_loss * self.denoising_loss_ratio  # i am not sure whether this is useful ?
            nll_loss = denoising_nll_loss * self.denoising_loss_ratio

        elif task_type == 'only_translation':
            net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                         "src_lengths": sample["net_input"]['src_lengths'],
                         "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
            net_output = model(**net_input)
            translation_loss, translation_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = translation_loss
            nll_loss = translation_nll_loss

        else:

            if self.denoising_loss_type == 'round_by_step':

                self.round = (self.round + 1) % 2

                if self.round == 0:  # we calculate translation loss
                    net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                                 "src_lengths": sample["net_input"]['src_lengths'],
                                 "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                    net_output = model(**net_input)
                    translation_loss, translation_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                    loss = translation_loss
                    nll_loss = translation_nll_loss

                else:
                    net_input = {"src_tokens": sample["net_input"]['noisy_target'],
                                 "src_lengths": sample["net_input"]['noisy_target_lengths'],
                                 "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                    net_output = model(**net_input)
                    denoising_loss, denoising_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                    loss = denoising_loss * self.denoising_loss_ratio  # i am not sure whether this is useful ?
                    nll_loss = denoising_nll_loss * self.denoising_loss_ratio

            elif self.denoising_loss_type == 'simultaneous':
                translation_net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                                         "src_lengths": sample["net_input"]['src_lengths'],
                                         "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                translation_net_output = model(**translation_net_input)
                translation_loss, translation_nll_loss = self.compute_loss(model, translation_net_output, sample,
                                                                           reduce=reduce)

                denoising_net_input = {"src_tokens": sample["net_input"]['noisy_target'],
                                       "src_lengths": sample["net_input"]['noisy_target_lengths'],
                                       "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                denoising_net_output = model(**denoising_net_input)
                denoising_loss, denoising_nll_loss = self.compute_loss(model, denoising_net_output, sample, reduce=reduce)

                loss = translation_loss + self.denoising_loss_ratio * denoising_loss
                nll_loss = translation_nll_loss + self.denoising_loss_ratio * denoising_nll_loss

            elif self.denoising_loss_type == 'round_by_epoch' and epoch is not None:

                if epoch % 2 == 1:
                    net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                                 "src_lengths": sample["net_input"]['src_lengths'],
                                 "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                    net_output = model(**net_input)
                    translation_loss, translation_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                    loss = translation_loss
                    nll_loss = translation_nll_loss

                else:
                    net_input = {"src_tokens": sample["net_input"]['noisy_target'],
                                 "src_lengths": sample["net_input"]['noisy_target_lengths'],
                                 "prev_output_tokens": sample["net_input"]['prev_output_tokens']}
                    net_output = model(**net_input)
                    denoising_loss, denoising_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                    loss = denoising_loss * self.denoising_loss_ratio  # i am not sure whether this is useful ?
                    nll_loss = denoising_nll_loss * self.denoising_loss_ratio

            else:
                # sample ratio ? like 10: 1 or others
                pass

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "translation_loss": utils.item(translation_loss.data) if reduce else translation_loss.data,
            "translation_nll_loss": utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            "denoising_loss": utils.item(denoising_loss.data) if reduce else denoising_loss.data,
            "denoising_nll_loss": utils.item(denoising_nll_loss.data) if reduce else denoising_nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training.
        We append translation and denoising loss separately
        """
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        translation_loss_sum = sum(log.get("translation_loss", 0) for log in logging_outputs)
        translation_nll_loss_sum = sum(log.get("translation_nll_loss", 0) for log in logging_outputs)
        denoising_loss_sum = sum(log.get("denoising_loss", 0) for log in logging_outputs)
        denoising_nll_loss_sum = sum(log.get("denoising_nll_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "translation_loss", translation_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "translation_nll_loss", translation_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "denoising_loss", denoising_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "denoising_nll_loss", denoising_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_derived(
            "translation_ppl", lambda meters: utils.get_perplexity(meters["translation_nll_loss"].avg)
        )
        metrics.log_derived(
            "denoising_ppl", lambda meters: utils.get_perplexity(meters["denoising_nll_loss"].avg)
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
