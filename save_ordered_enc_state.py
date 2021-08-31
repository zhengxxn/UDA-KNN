# !/usr/bin/env python3 -u
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

    # as we want to save the original order of the data (idx of each sent)
    # to achieve this, we first record the idx in numpy array for each sent
    # then we iterate over the sent record dict, to get the true idx

    sent_record_dict = {}

    # --- check save data store , add by
    # the length of datastore is same as the sentence count of dataset
    import numpy as np
    if args.dstore_fp16:
        print('Saving fp16')
        enc_output_np = np.memmap(args.enc_output_save_dir + '/enc_output.npy', dtype=np.float16, mode='w+',
                                  shape=(args.enc_dstore_size, args.enc_state_dim))

    else:
        print('Saving fp32')
        enc_output_np = np.memmap(args.enc_output_save_dir + '/enc_output.npy', dtype=np.float32, mode='w+',
                                  shape=(args.enc_dstore_size, args.enc_state_dim))

    dstore_idx = 0
    # --- end
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

        log_outputs = []
        with torch.no_grad():

            model.eval()
            for i, sample in enumerate(progress):

                sample = utils.move_to_cuda(sample) if use_cuda else sample

                if args.save_denoising_feature:
                    sample['net_input']['src_tokens'] = sample['net_input']['noisy_target']
                    sample['net_input']['src_lengths'] = sample['net_input']['noisy_target_lengths']

                # TODO: this get enc hidden state, so we should make a modification here
                if not args.activate_adapter:
                    enc_output = task.forward_and_get_enc_hidden_state(sample, model)  # [B, H]
                else:
                    enc_output = task.forward_and_get_enc_hidden_state(sample, model,
                                                                       activate_adapter=args.activate_adapter)  # [B, H]
                # get useful parameters
                batch_size = enc_output.size(0)

                # we use this to save sent record id
                target_id = sample['id'].tolist()
                cur_dstore_idx = dstore_idx

                for idx_of_id, sent_id in enumerate(target_id):
                    sent_record_dict[sent_id] = cur_dstore_idx
                    cur_dstore_idx += 1

                current_batch_count = batch_size

                # reduce, if too long
                if dstore_idx + current_batch_count > args.enc_dstore_size:
                    reduce_size = args.enc_dstore_size - dstore_idx
                    enc_output = enc_output[:reduce_size]
                else:
                    reduce_size = current_batch_count

                if args.dstore_fp16:
                    enc_output_np[dstore_idx: reduce_size + dstore_idx] = enc_output.detach().cpu().numpy().astype(
                        np.float16)

                else:
                    enc_output_np[dstore_idx: reduce_size + dstore_idx] = enc_output.detach().cpu().numpy().astype(
                        np.float32)

                dstore_idx += reduce_size

                # print(dstore_idx)
                progress.log({'dstore_size': dstore_idx}, step=i)
                if dstore_idx > args.enc_dstore_size:
                    print('much more than dstore size break')
                    break

                elif dstore_idx == args.enc_dstore_size:
                    print('just fill')
            # -------- end, by

        # save the order key and value
        print('save order key and value')
        print(dstore_idx)

        reordered_idx = []
        sent_id_list = list(sent_record_dict.keys())
        sent_id_list.sort()

        # if 17222 in sent_id_list:
        #     print("id 17222 in list")
        #     sent_id_list.remove(17222)

        for sent_id in sent_id_list:
            reordered_idx.append(sent_record_dict[sent_id])

        print(len(reordered_idx))
        ordered_enc_output_np = np.memmap(args.enc_output_save_dir + '/ordered_enc_output.npy', dtype=np.float16, mode='w+',
                                          shape=(len(reordered_idx), args.enc_state_dim))

        ordered_enc_output_np[:] = enc_output_np[reordered_idx]


def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
