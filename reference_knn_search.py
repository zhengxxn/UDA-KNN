#!/usr/bin/env python3 -u
# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


# ------ add by
# this script is implemented based on validate.py, and refers to the implementation of knnlm
# we only need to go through the dataset like in training, and save the datastore
# ------

def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion, we do not need this, so remove it, by
    # criterion = task.build_criterion(model_args)
    # criterion.eval()
    data_idx = 1
    for subset in args.valid_subset.split(","):
        try:
            task.args.required_seq_len_multiple = 1
            task.args.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        reference_knn_record = {}
        tgt_dict = task.target_dictionary
        log_outputs = []
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                knn_prob, knn_lambda, knn_dists, knn_index = task.forward_and_get_knn_record(sample, model)  # [B, T, H]
                target = sample['target']  # [B, T]

                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]
                target_length = torch.sum(target_mask, dim=1).int().tolist()  # [B]

                batch_size = target.size(0)

                for j in range(0, batch_size):
                    reference_knn_record[sample["id"].tolist()[j]] = \
                        {
                            'reference': tgt_dict.string(target[j][:target_length[j]], return_list=True),
                            'knn_index': knn_index[j][: target_length[j]].tolist(),  # [L, K]
                            'knn_distance': knn_dists[j][:target_length[j]].tolist(),  # [L, K]
                        }

        with open(args.knn_record_file, 'wb') as f:
            import pickle
            pickle.dump(reference_knn_record, f)


def cli_main():

    parser = options.get_reference_knn_search_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_reference_knn_search_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
