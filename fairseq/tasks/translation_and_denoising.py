# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
import torch
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    TranslationAndDenoisingDataset,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks.translation import load_langpair_dataset

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


def load_langpair_with_denoising_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        mask_idx=-1,  # we append denoising parameters from here
        mask_whole_words=False,
        mask_ratio=0.0,
        random_ratio=0.0,
        insert_ratio=0.0,
        rotate_ratio=0.0,
        permute_ratio=0.0,
        permute_sentence_ratio=0.0,
        item_transform_func=None,
        replace_length=None,
        mask_length=None,
        poisson_lambda=0.0,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return TranslationAndDenoisingDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        mask_idx=mask_idx,  # we append denoising parameters from here
        mask_whole_words=mask_whole_words,
        mask_ratio=mask_ratio,
        random_ratio=random_ratio,
        insert_ratio=insert_ratio,
        rotate_ratio=rotate_ratio,
        permute_ratio=permute_ratio,
        permute_sentence_ratio=permute_sentence_ratio,
        item_transform_func=item_transform_func,
        replace_length=replace_length,
        mask_length=mask_length,
        poisson_lambda=poisson_lambda,
    )


@register_task("translation_and_denoising")
class TranslationAndDenoisingTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    and Combined with Denoising Task.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off

        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories. ')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

        # for simplicity, we copy the denoising arguments from denoising.py to here, and modify the arguments' name

        parser.add_argument("--denoising-mask-ratio", default=0.0, type=float,
                            help="fraction of words/subwords that will be masked", )

        parser.add_argument("--denoising-mask-random-ratio", default=0.0, type=float,
                            help="instead of using [MASK], use random token this often", )

        parser.add_argument("--denoising-insert-ratio", default=0.0, type=float,
                            help="insert this percentage of additional random tokens", )

        parser.add_argument("--denoising-permute-ratio", default=0.0, type=float,
                            help="take this proportion of subwords and permute them", )

        parser.add_argument("--denoising-rotate-ratio", default=0.0, type=float,
                            help="rotate this proportion of inputs", )

        parser.add_argument("--denoising-poisson-lambda", default=3.0, type=float,
                            help="randomly shuffle sentences for this proportion of inputs", )

        parser.add_argument("--denoising-permute-sentences-ratio", default=0.0, type=float,
                            help="shuffle this proportion of sentences in all inputs", )

        parser.add_argument("--denoising-mask-length", default="subword", type=str,
                            choices=["subword", "word", "span-poisson"],
                            help="mask length to choose",
                            )

        parser.add_argument("--denoising-replace-length", default=-1, type=int,
                            help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)", )

        # end
        parser.add_argument("--train-task", type=str, default="only_translation",
                            help="only_translation, denoising_approximate, translation_and_denoising")
        parser.add_argument("--validation-task", type=str, default="only_translation",
                            help="only_translation, denoising_approximate, translation_and_denoising")

        #
        parser.add_argument("--select-last-as-mask", default=False, action="store_true",
                            help="we change the last token in vocab to mask")

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.current_epoch = 0
        self.train_task = args.train_task
        self.validation_task = args.validation_task
        self.select_last_as_mask = args.select_last_as_mask
        # copy from denoising task
        # we suppose that denoising task is performed on target
        # self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang))
        )

        # if we want to simulate Denoising and Translation in one model, the embedding must share
        # add by
        if args.select_last_as_mask:
            src_mask = len(src_dict.symbols) - 1
            tgt_mask = len(tgt_dict.symbols) - 1
        else:
            src_mask = src_dict.add_symbol("<mask>")
            tgt_mask = tgt_dict.add_symbol("<mask>")

        assert src_mask == tgt_mask
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        if not hasattr(args, "denoising_shuffle_instance"):
            args.shuffle_instance = False

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.denoising_mask_length != "subword"
            else None
        )

        print("load dataset: ", split)

        # for simplicity, we distinguish different dataset by split here
        # if split == 'train':
        self.datasets[split] = load_langpair_with_denoising_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
            mask_idx=self.tgt_dict.indices['<mask>'] if not self.select_last_as_mask else len(self.tgt_dict.symbols) - 1,
            mask_whole_words=mask_whole_words,
            mask_ratio=self.args.denoising_mask_ratio,
            random_ratio=self.args.denoising_mask_random_ratio,
            insert_ratio=self.args.denoising_insert_ratio,
            rotate_ratio=self.args.denoising_rotate_ratio,
            permute_ratio=self.args.denoising_permute_ratio,
            permute_sentence_ratio=self.args.denoising_permute_sentences_ratio,
            item_transform_func=None,
            replace_length=self.args.denoising_replace_length,
            mask_length=self.args.denoising_mask_length,
            poisson_lambda=self.args.denoising_poisson_lambda,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def forward_and_get_hidden_state_step(self, sample, model, need_self_attn=False, activate_adapter=False):
        # add by
        # forward the model with the sample, and get the decoder hidden state used for datastore
        # and we only need the feature
        decoder_output, extra = model(src_tokens=sample['net_input']['src_tokens'],
                                      src_lengths=sample['net_input']['src_lengths'],
                                      prev_output_tokens=sample['net_input']['prev_output_tokens'],
                                      return_all_hiddens=False,
                                      features_only=True,
                                      need_self_attention_weights=need_self_attn,
                                      activate_adapter=activate_adapter)
        return decoder_output, extra

    def forward_and_get_enc_hidden_state(self, sample, model, activate_adapter=False):

        enc_output = model.encoder(src_tokens=sample['net_input']['src_tokens'],
                                   src_lengths=sample['net_input']['src_lengths'],
                                   activate_adapter=activate_adapter)

        enc_last_hidden = enc_output[0].transpose(0, 1)  # [B, L, H]
        enc_pad_mask = enc_output[1]  # [B, L]
        enc_last_hidden = enc_last_hidden.masked_fill(mask=enc_pad_mask.unsqueeze(-1), value=0)

        enc_last_hidden = enc_last_hidden.sum(dim=1)
        enc_last_hidden = enc_last_hidden / sample['net_input']['src_lengths'].unsqueeze(-1).float()
        return enc_last_hidden


    def forward_and_get_knn_record(self, sample, model):
        # make sure this is with knn search model
        if model.decoder.label_count_as_feature:
            decoder_output, extra, knn_prob, knn_lambda, knn_dists, knn_index, label_counts = model(
                src_tokens=sample['net_input']['src_tokens'],
                src_lengths=sample['net_input']['src_lengths'],
                prev_output_tokens=sample['net_input']['prev_output_tokens'],
                return_all_hiddens=False,
                features_only=True)
            return knn_prob, knn_lambda, knn_dists, knn_index, label_counts
        else:
            decoder_output, extra, knn_prob, knn_lambda, knn_dists, knn_index = model(
                src_tokens=sample['net_input']['src_tokens'],
                src_lengths=sample['net_input']['src_lengths'],
                prev_output_tokens=sample['net_input']['prev_output_tokens'],
                return_all_hiddens=False,
                features_only=True)
            return knn_prob, knn_lambda, knn_dists, knn_index

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model,
                                                          sample,
                                                          task_type=self.train_task,
                                                          epoch=self.current_epoch,
                                                          num_updates=update_num,
                                                          is_train=True)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):

        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, task_type=self.validation_task)

        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output

    def begin_epoch(self, epoch, model):

        self.current_epoch += 1

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        sample["net_input"].pop("noisy_target", None)
        sample["net_input"].pop("noisy_target_lengths", None)

        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
