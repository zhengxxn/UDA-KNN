# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from .label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion("label_smoothed_cross_entropy_with_denoising_approximate")
class LabelSmoothedCrossEntropyCriterionWithDenoisingApproximate(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 denoising_approximate_loss_type='mse',
                 denoising_approximate_loss_ratio=0.0,
                 denoising_start_epoch=1,
                 # only_record_denoising_loss=False,
                 update_denoising_with_adapter=False,
                 denoising_approximate_loss_ratio_begin=None,
                 negative_samples=0,
                 marginal_value=0.0,
                 marginal_loss_ratio=1.0,
                 marginal_loss_start_step=0):

        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)

        self.denoising_approximate_loss_type = denoising_approximate_loss_type

        # we set denoising approximate loss func here
        self.denoising_approximate_loss_func = None
        if self.denoising_approximate_loss_type == 'mse':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'marginal_mse':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'translation_marginal_mse':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'denoising_marginal_mse':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'mix_marginal_mse':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'mix_marginal_and_min_distance':
            self.denoising_approximate_loss_func = nn.MSELoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'cos':
            self.denoising_approximate_loss_func = nn.CosineEmbeddingLoss(reduction='none')
        elif self.denoising_approximate_loss_type == 'kl':
            self.denoising_approximate_loss_func = nn.KLDivLoss(reduction='none')

        self.denoising_approximate_loss_ratio = denoising_approximate_loss_ratio
        self.denoising_start_epoch = denoising_start_epoch

        # self.only_record_denoising_loss = only_record_denoising_loss
        self.update_denoising_with_adapter = update_denoising_with_adapter
        self.denoising_approximate_loss_ratio_begin = denoising_approximate_loss_ratio_begin

        self.negative_samples = negative_samples
        self.marginal_value = marginal_value
        self.marginal_loss_ratio = marginal_loss_ratio
        self.marginal_loss_start_step = marginal_loss_start_step

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)

        parser.add_argument(
            "--denoising-approximate-loss-type",
            default="mse",
            type=str,
            help="mse, kl, cos"
        )

        parser.add_argument(
            "--denoising-approximate-loss-ratio",
            default=1.0,
            type=float,
            metavar="D",
            help="weight for the denoising loss",
        )

        parser.add_argument(
            "--denoising-approximate-loss-ratio-begin",
            default=None,
            type=float,
        )

        # just deprecated for simplicity
        # parser.add_argument(
        #     "--denoising-approximate-loss-temperature",
        #     default=1.0,
        #     type=float,
        #     help="if we use kl loss, this may used"
        # )

        parser.add_argument(
            "--denoising-start-epoch",
            default=1,
            type=int,
            help="when to start denoising task"
        )

        # parser.add_argument(
        #     "--only-record-denoising-loss",
        #     default=False,
        #     action="store_true",
        #     help="if this is set, we do not update model with denoising loss, but we will record it for analysis"
        # )

        parser.add_argument(
            "--update-denoising-with-adapter",
            default=False,
            action="store_true"
        )

        parser.add_argument(
            "--negative-samples",
            default=0,
            type=int,
        )

        parser.add_argument(
            "--marginal-value",
            default=0.0,
            type=float
        )

        parser.add_argument(
            "--marginal-loss-ratio",
            default=1.0,
            type=float
        )

        parser.add_argument(
            "--marginal-loss-start-step",
            default=0,
            type=int,
        )

    def forward(self, model, sample, reduce=True, task_type='only_translation', epoch=None, num_updates=None,
                is_train=False):

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
        denoising_approximate_loss = torch.zeros(1)
        mix_marginal_loss = torch.zeros(1)
        approximate_loss = torch.zeros(1)

        if task_type == 'only_translation' or (epoch is not None and epoch < self.denoising_start_epoch):
            # this is same as translation task

            net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                         "src_lengths": sample["net_input"]['src_lengths'],
                         "prev_output_tokens": sample["net_input"]['prev_output_tokens']}

            net_output = model(**net_input)
            translation_loss, translation_nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = translation_loss
            nll_loss = translation_nll_loss

        else:
            # task_type == "denoising_approximate" or "translation_and_denoising"

            translation_net_input = {"src_tokens": sample["net_input"]['src_tokens'],
                                     "src_lengths": sample["net_input"]['src_lengths'],
                                     "prev_output_tokens": sample["net_input"]['prev_output_tokens']}

            translation_pre_softmax_representation = None
            translation_probs = None

            # if we only use translation output to be the target of denoising task
            if task_type == "denoising_approximate" and is_train:

                model.eval()
                with torch.no_grad():

                    translation_net_output = model(**translation_net_input)
                    translation_pre_softmax_representation = translation_net_output[1]["last_hidden"].detach()
                    translation_loss, translation_nll_loss, translation_probs = self.compute_loss_and_return_probs(
                        model,
                        translation_net_output,
                        sample,
                        reduce=reduce)
                    translation_probs = translation_probs.detach()

                model.train()

            elif task_type == "denoising_approximate":  # in validation

                translation_net_output = model(**translation_net_input)
                translation_pre_softmax_representation = translation_net_output[1]["last_hidden"].detach()

            else:  # translation_and_denoising
                print("translation and approximate")
                translation_net_output = model(**translation_net_input)
                translation_pre_softmax_representation = translation_net_output[1]["last_hidden"].detach()
                translation_loss, translation_nll_loss = self.compute_loss(model, translation_net_output, sample,
                                                                           reduce=reduce)

            denoising_net_input = {"src_tokens": sample["net_input"]['noisy_target'],
                                   "src_lengths": sample["net_input"]['noisy_target_lengths'],
                                   "prev_output_tokens": sample["net_input"]['prev_output_tokens'],
                                   "activate_adapter": self.update_denoising_with_adapter}
            denoising_net_output = model(**denoising_net_input)

            # print(sample["net_input"]["noisy_target"][:5])
            # print(sample["target"][:5])
            target = sample["target"]  # [B, S]
            non_pad_mask = target.ne(self.padding_idx)  # [B, S], 1 for not pad, 0 for pad
            non_pad_mask = non_pad_mask.view(-1, 1)  # [B * S, 1]
            non_pad_idx = non_pad_mask.nonzero(as_tuple=True)[0]  # [ Select Size, 1]

            if self.denoising_approximate_loss_type == 'cos':
                pass
                # [B * S, H]
                # denoising_pre_softmax_representation = denoising_net_output[1]["last_hidden"]  # [B, S, H]
                #
                # translation_pre_softmax_representation = \
                #     translation_pre_softmax_representation.reshape(-1, translation_pre_softmax_representation.size(-1))
                # denoising_pre_softmax_representation = \
                #     denoising_pre_softmax_representation.reshape(-1, denoising_pre_softmax_representation.size(-1))
                #
                # translation_pre_softmax_representation = translation_pre_softmax_representation.index_select(dim=0,
                #                                                                                              index=non_pad_idx)
                # denoising_pre_softmax_representation = denoising_pre_softmax_representation.index_select(dim=0,
                #                                                                                          index=non_pad_idx)

                # input / target
                # denoising_approximate_loss = self.denoising_approximate_loss_func(denoising_pre_softmax_representation,
                #                                                                   translation_pre_softmax_representation,
                #                                                                   torch.ones(1).to(
                #                                                                       translation_pre_softmax_representation.device))  # [ not pad size]
                # print(denoising_approximate_loss.size())
                # print(denoising_approximate_loss[:50])
                # denoising_approximate_loss = torch.exp(denoising_approximate_loss)  # [not pad size]
                # print(denoising_approximate_loss[:50])

                # we apply scaling loss here for not overflow the fp16
                # if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                #         and self.denoising_approximate_loss_ratio_begin is not None:
                #     denoising_approximate_loss = self.denoising_approximate_loss_ratio_begin * denoising_approximate_loss
                # else:
                #     denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss
                #
                # denoising_approximate_loss = denoising_approximate_loss.sum(dim=-1)  # [1]

            elif self.denoising_approximate_loss_type == 'mse':

                denoising_pre_softmax_representation = denoising_net_output[1]["last_hidden"]  # [B, S, H]

                translation_pre_softmax_representation = \
                    translation_pre_softmax_representation.reshape(-1, translation_pre_softmax_representation.size(-1))
                denoising_pre_softmax_representation = \
                    denoising_pre_softmax_representation.reshape(-1, denoising_pre_softmax_representation.size(-1))

                translation_pre_softmax_representation = translation_pre_softmax_representation.index_select(dim=0,
                                                                                                             index=non_pad_idx)
                denoising_pre_softmax_representation = denoising_pre_softmax_representation.index_select(dim=0,
                                                                                                         index=non_pad_idx)

                # input / target
                denoising_approximate_loss = self.denoising_approximate_loss_func(denoising_pre_softmax_representation,
                                                                                  translation_pre_softmax_representation)  # [ not pad size, H]

                # we apply scaling loss here for not overflow the fp16
                if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                        and self.denoising_approximate_loss_ratio_begin is not None:
                    denoising_approximate_loss = self.denoising_approximate_loss_ratio_begin * denoising_approximate_loss
                else:
                    denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss

                denoising_approximate_loss = denoising_approximate_loss.sum(dim=-1)  # [ not pad size ]
                denoising_approximate_loss = denoising_approximate_loss.sum(dim=-1)  # [1]

            elif self.denoising_approximate_loss_type == 'kl':

                denoising_lprobs = self.get_lprobs(model, denoising_net_output, sample)  # [B*S, V]

                translation_probs = translation_probs.index_select(dim=0, index=non_pad_idx)  # [ not pad size, V]
                denoising_lprobs = denoising_lprobs.index_select(dim=0, index=non_pad_idx)  # [ not pad size, V]

                denoising_approximate_loss = self.denoising_approximate_loss_func(denoising_lprobs,
                                                                                  translation_probs)  # [ not pad size, V]
                # exit(0)

                # we apply scaling loss here for not overflow the fp16
                if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                        and self.denoising_approximate_loss_ratio_begin is not None:
                    denoising_approximate_loss = self.denoising_approximate_loss_ratio_begin * denoising_approximate_loss
                else:
                    denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss

                denoising_approximate_loss = denoising_approximate_loss.sum(-1)  # [not pad size]
                # TODO, this may cause overflow problem in fp16, if denoising approximate loss is too large
                # TODO, we may move the scaled by loss ratio before summation
                # print(denoising_approximate_loss[:10])
                denoising_approximate_loss = denoising_approximate_loss.sum(-1)
                # print(denoising_approximate_loss)
                # denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss

            elif self.denoising_approximate_loss_type == 'translation_marginal_mse':
                # for each translation representation, we sample some negative samples from
                # denoising representations and apply marginal loss function:
                # suppose the positive l2 distance is d1, and the negative l2 distances are d2, d3, d4, ...
                # for each negative distance, we compute its loss, that is:
                #   if (d2 < d1 + d_gap), that we have loss d1 + d_gap - d2, which means d2 is small and we want to
                # make it large

                denoising_pre_softmax_representation = denoising_net_output[1]["last_hidden"]  # [B, S, H]

                translation_pre_softmax_representation = \
                    translation_pre_softmax_representation.reshape(-1, translation_pre_softmax_representation.size(-1))
                denoising_pre_softmax_representation = \
                    denoising_pre_softmax_representation.reshape(-1, denoising_pre_softmax_representation.size(-1))

                translation_pre_softmax_representation = translation_pre_softmax_representation.index_select(dim=0,
                                                                                                             index=non_pad_idx)
                denoising_pre_softmax_representation = denoising_pre_softmax_representation.index_select(dim=0,
                                                                                                         index=non_pad_idx)

                # we sample the negative representations here ( from denoising representation )
                total_range = denoising_pre_softmax_representation.size(0)
                negative_index = torch.randint(0, total_range, (total_range, self.negative_samples)).view(
                    total_range * self.negative_samples).to(denoising_pre_softmax_representation.device)
                negative_denoising_representations = denoising_pre_softmax_representation.index_select(
                    index=negative_index, dim=0).view(total_range, self.negative_samples,
                                                      denoising_pre_softmax_representation.size(-1))

                positive_distance = self.denoising_approximate_loss_func(denoising_pre_softmax_representation,
                                                                         translation_pre_softmax_representation)
                positive_distance = positive_distance.sum(dim=-1)  # [ not pad size ]

                negative_distances = self.denoising_approximate_loss_func(negative_denoising_representations,
                                                                          translation_pre_softmax_representation.unsqueeze(
                                                                              1).expand_as(
                                                                              negative_denoising_representations))
                negative_distances = negative_distances.sum(dim=-1)  # [not pad size, neg samples]
                positive_distance = positive_distance.unsqueeze(-1)  # [not pad size, 1]
                denoising_approximate_loss = positive_distance - negative_distances  # [not pad size, neg samples]
                denoising_approximate_loss = denoising_approximate_loss + self.marginal_value

                denoising_approximate_loss[denoising_approximate_loss < 0.0] = 0.0
                denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss
                denoising_approximate_loss = denoising_approximate_loss.sum()
                # print(torch.mean(sample["net_input"]['noisy_target_lengths'].float()))
                # print(denoising_approximate_loss)

            elif self.denoising_approximate_loss_type == 'denoising_marginal_mse':

                denoising_pre_softmax_representation = denoising_net_output[1]["last_hidden"]  # [B, S, H]

                translation_pre_softmax_representation = \
                    translation_pre_softmax_representation.reshape(-1, translation_pre_softmax_representation.size(-1))
                denoising_pre_softmax_representation = \
                    denoising_pre_softmax_representation.reshape(-1, denoising_pre_softmax_representation.size(-1))

                translation_pre_softmax_representation = translation_pre_softmax_representation.index_select(dim=0,
                                                                                                             index=non_pad_idx)
                denoising_pre_softmax_representation = denoising_pre_softmax_representation.index_select(dim=0,
                                                                                                         index=non_pad_idx)

                # we sample the negative representations here ( from translation representation)
                total_range = denoising_pre_softmax_representation.size(0)
                negative_index = torch.randint(0, total_range, (total_range, self.negative_samples)).view(
                    total_range * self.negative_samples).to(translation_pre_softmax_representation.device)
                negative_translation_representations = translation_pre_softmax_representation.index_select(
                    index=negative_index, dim=0).view(total_range, self.negative_samples,
                                                      translation_pre_softmax_representation.size(-1))

                positive_distance = self.denoising_approximate_loss_func(denoising_pre_softmax_representation,
                                                                         translation_pre_softmax_representation)
                positive_distance = positive_distance.sum(dim=-1)  # [ not pad size ]

                negative_distances = self.denoising_approximate_loss_func(
                    denoising_pre_softmax_representation.unsqueeze(1).expand_as(negative_translation_representations),
                    negative_translation_representations)
                negative_distances = negative_distances.sum(dim=-1)  # [not pad size, neg samples]

                positive_distance = positive_distance.unsqueeze(-1)  # [not pad size, 1]

                denoising_approximate_loss = positive_distance + self.marginal_value - negative_distances  # [not pad size, neg samples]
                denoising_approximate_loss[denoising_approximate_loss < 0.0] = 0.0
                denoising_approximate_loss = self.denoising_approximate_loss_ratio * denoising_approximate_loss
                denoising_approximate_loss = denoising_approximate_loss.sum()

            elif self.denoising_approximate_loss_type == 'mix_marginal_mse' or self.denoising_approximate_loss_type == 'mix_marginal_and_min_distance':

                denoising_pre_softmax_representation = denoising_net_output[1]["last_hidden"]  # [B, S, H]

                translation_pre_softmax_representation = \
                    translation_pre_softmax_representation.reshape(-1, translation_pre_softmax_representation.size(-1))
                denoising_pre_softmax_representation = \
                    denoising_pre_softmax_representation.reshape(-1, denoising_pre_softmax_representation.size(-1))

                translation_pre_softmax_representation = translation_pre_softmax_representation.index_select(dim=0,
                                                                                                             index=non_pad_idx)
                denoising_pre_softmax_representation = denoising_pre_softmax_representation.index_select(dim=0,
                                                                                                         index=non_pad_idx)

                # we sample the negative representations here ( from denoising representation )
                total_range = denoising_pre_softmax_representation.size(0)
                denoising_negative_index = torch.randint(0, total_range, (total_range, self.negative_samples)).view(
                    total_range * self.negative_samples).to(denoising_pre_softmax_representation.device)
                denoising_negative_representations = denoising_pre_softmax_representation.index_select(
                    index=denoising_negative_index, dim=0).view(total_range, self.negative_samples,
                                                                denoising_pre_softmax_representation.size(-1))

                positive_distance = self.denoising_approximate_loss_func(denoising_pre_softmax_representation,
                                                                         translation_pre_softmax_representation)
                positive_distance = positive_distance.sum(dim=-1)  # [ not pad size ]
                positive_distance = positive_distance.unsqueeze(-1)  # [not pad size, 1]

                # for each translation representation, we sample some negative representations from denoising samples
                each_translation_negative_distances = self.denoising_approximate_loss_func(
                    denoising_negative_representations,
                    translation_pre_softmax_representation.unsqueeze(
                        1).expand_as(
                        denoising_negative_representations))

                each_translation_negative_distances = each_translation_negative_distances.sum(
                    dim=-1)  # [not pad size, neg samples]
                each_translation_denoising_approximate_loss = positive_distance * self.marginal_value - each_translation_negative_distances  # [not pad size, neg samples]
                # each_translation_denoising_approximate_loss = each_translation_denoising_approximate_loss + self.marginal_value

                each_translation_denoising_approximate_loss[each_translation_denoising_approximate_loss < 0.0] = 0.0

                if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                        and self.denoising_approximate_loss_ratio_begin is not None:
                    each_translation_denoising_approximate_loss = self.denoising_approximate_loss_ratio_begin * each_translation_denoising_approximate_loss
                else:
                    each_translation_denoising_approximate_loss = self.denoising_approximate_loss_ratio * each_translation_denoising_approximate_loss

                each_translation_denoising_approximate_loss = each_translation_denoising_approximate_loss.sum()

                # sample translation neg samples for each denoising representation
                # TODO, we commented out these code here, for not using them currently.
                # translation_negative_index = torch.randint(0, total_range, (total_range, self.negative_samples)).view(
                #     total_range * self.negative_samples).to(translation_pre_softmax_representation.device)
                # translation_negative_representations = translation_pre_softmax_representation.index_select(
                #     index=translation_negative_index, dim=0).view(total_range, self.negative_samples,
                #                                                   translation_pre_softmax_representation.size(-1))
                #
                # each_denoising_negative_distances = self.denoising_approximate_loss_func(
                #     denoising_pre_softmax_representation.unsqueeze(1).expand_as(translation_negative_representations),
                #     translation_negative_representations)
                # each_denoising_negative_distances = each_denoising_negative_distances.sum(dim=-1)  # [not pad size, neg samples]
                #
                # each_denoising_denoising_approximate_loss = positive_distance * self.marginal_value - each_denoising_negative_distances  # [not pad size, neg samples]
                # each_denoising_denoising_approximate_loss[each_denoising_denoising_approximate_loss < 0.0] = 0.0
                #
                # if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                #         and self.denoising_approximate_loss_ratio_begin is not None:
                #     each_denoising_denoising_approximate_loss = self.denoising_approximate_loss_ratio_begin * each_denoising_denoising_approximate_loss
                # else:
                #     each_denoising_denoising_approximate_loss = self.denoising_approximate_loss_ratio * each_denoising_denoising_approximate_loss
                #
                # each_denoising_denoising_approximate_loss = each_denoising_denoising_approximate_loss.sum()

                # denoising_approximate_loss = each_translation_denoising_approximate_loss + each_denoising_denoising_approximate_loss
                denoising_approximate_loss = each_translation_denoising_approximate_loss
                # for record
                mix_marginal_loss = denoising_approximate_loss.detach() * self.marginal_loss_ratio

                if self.denoising_approximate_loss_type == 'mix_marginal_and_min_distance':

                    if ((epoch is not None and epoch == 1) and (num_updates is not None and num_updates <= 5000)) \
                            and self.denoising_approximate_loss_ratio_begin is not None:
                        positive_distance = self.denoising_approximate_loss_ratio_begin * positive_distance
                    else:
                        positive_distance = self.denoising_approximate_loss_ratio * positive_distance

                    positive_distance = positive_distance.sum()
                    approximate_loss = positive_distance.detach()

                    if num_updates is not None and num_updates >= self.marginal_loss_start_step:
                        denoising_approximate_loss = self.marginal_loss_ratio * denoising_approximate_loss + positive_distance
                    else:
                        denoising_approximate_loss = positive_distance

            if task_type == "denoising_approximate":
                loss = denoising_approximate_loss
            else:
                loss = translation_loss + denoising_approximate_loss
                nll_loss = translation_nll_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "translation_loss": utils.item(translation_loss.data) if reduce else translation_loss.data,
            "translation_nll_loss": utils.item(translation_nll_loss.data) if reduce else translation_nll_loss.data,
            "denoising_approximate_loss": utils.item(
                denoising_approximate_loss.data) if reduce else denoising_approximate_loss.data,
            "mix_marginal_loss": utils.item(mix_marginal_loss.data) if reduce else mix_marginal_loss.data,
            "approximate_loss": utils.item(approximate_loss.data) if reduce else approximate_loss.data,
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
        denoising_approximate_loss_sum = sum(log.get("denoising_approximate_loss", 0) for log in logging_outputs)
        approximate_loss_sum = sum(log.get("approximate_loss", 0) for log in logging_outputs)
        mix_marginal_loss_sum = sum(log.get("mix_marginal_loss", 0) for log in logging_outputs)

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
            "denoising_approximate_loss", denoising_approximate_loss_sum / sample_size / math.log(2), sample_size,
            round=3
        )
        metrics.log_scalar(
            "approximate_loss", approximate_loss_sum / sample_size / math.log(2), sample_size,
            round=3
        )

        metrics.log_scalar(
            "mix_marginal_loss", mix_marginal_loss_sum / sample_size / math.log(2), sample_size,
            round=3
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_derived(
            "translation_ppl", lambda meters: utils.get_perplexity(meters["translation_nll_loss"].avg)
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

    def get_lprobs_and_target_and_probs(self, model, net_output, sample):

        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = torch.log(probs)

        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), probs.view(-1, probs.size(-1))

    def compute_loss_and_return_probs(self, model, net_output, sample, reduce=True):

        lprobs, target, probs = self.get_lprobs_and_target_and_probs(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, probs

    def get_lprobs(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        return lprobs

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
