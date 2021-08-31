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

    # print(models[0])
    # exit(0)
    # Print a1rgs
    logger.info(model_args)

    # Build criterion, we do not need this, so remove it, by
    # criterion = task.build_criterion(model_args)
    # criterion.eval()
    if args.save_plain_text:
        # we use two lists to save the plain text, where the first list is used to locate the sample (sentence)
        # by index, the second list[dict] contains {start_idx, end_idx, src_tokens, trg_tokens} for each sample
        locate_dict = []
        sentences_array = []

        cur_sent_start_idx = 0

        try:
            src_dict = getattr(task, "source_dictionary", None)
        except NotImplementedError:
            src_dict = None
        tgt_dict = task.target_dictionary

    if args.save_source_empty_feature:
        src_dict = getattr(task, "source_dictionary", None)

    # --- check save data store , add by
    import numpy as np
    if args.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.dstore_size, 1))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='w+',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.dstore_size, 1))

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
                # print(sample['net_input']['src_tokens'])
                # print(sample['net_input']['prev_output_tokens'])
                # print(sample['target'])
                # exit(0)
                # -------- add by , we should go through the model with the sample and get the hidden state
                # so we append a forward_and_get_hidden_state_step method in Translation task
                # todo, here we want to record the attention information for each token, which maybe too large

                if args.save_denoising_feature:
                    sample['net_input']['src_tokens'] = sample['net_input']['noisy_target']
                    sample['net_input']['src_lengths'] = sample['net_input']['noisy_target_lengths']

                if args.save_source_empty_feature:
                    src_batch_size = sample['net_input']['src_tokens'].size(0)
                    sample['net_input']['src_tokens'] = torch.zeros((src_batch_size, 1)).fill_(src_dict.eos()).to(sample['target'])
                    sample['net_input']['src_lengths'] = torch.ones((src_batch_size, 1)).to(sample['target'])

                if not args.activate_adapter:
                    features, extra = task.forward_and_get_hidden_state_step(sample, model,
                                                                             need_self_attn=args.save_attn_weights, )
                    # activate_adapter=args.activate_adapter)  # [B, T, H]
                else:
                    features, extra = task.forward_and_get_hidden_state_step(sample, model,
                                                                             need_self_attn=args.save_attn_weights,
                                                                             activate_adapter=args.activate_adapter)

                cross_attn = extra['attn'][0]  # [B, tgt len, src len]
                self_attn = extra['self_attn']  # [B, tgt len. prev tgt len]

                source_length = sample['net_input']['src_lengths']  # [B]
                target = sample['target']  # [B, T]

                # get useful parameters
                batch_size = target.size(0)
                seq_len = target.size(1)
                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]

                target_length = torch.sum(target_mask, dim=1).int()  # [B]

                # remove the pad tokens and related hidden states
                target = target.view(batch_size * seq_len)
                target_mask = target_mask.view(batch_size * seq_len)

                non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                target = target.index_select(dim=0, index=non_pad_index)  # [n_count]

                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                # if save plain text
                if args.save_plain_text:
                    cur_token_idx = len(locate_dict)

                    # here, we save the sent index for each token in every sentence, for example
                    # target tokens [s1t1, s1t2, s1t3, s2t1, s2t2, s3t1, s3t2, s3t3, s3t4]
                    # sent index    [0,    0,    0,    1,    1,    2,    2,    2,    2   ]

                    cur_token_locate = torch.arange(cur_sent_start_idx, cur_sent_start_idx + batch_size).to(
                        target.device)  # [B]
                    cur_token_locate = cur_token_locate.unsqueeze(-1).expand(batch_size, seq_len).reshape(
                        batch_size * seq_len)  # [B, T_S]
                    cur_token_locate = cur_token_locate.index_select(dim=0, index=non_pad_index).tolist()  # [n_count]
                    locate_dict += cur_token_locate
                    cur_sent_start_idx = cur_sent_start_idx + batch_size

                    for j in range(0, batch_size):
                        sentences_array.append(
                            {
                                'start_idx': cur_token_idx,
                                'end_idx': cur_token_idx + target_length[j].item() - 1,
                                'len': target_length[j].item(),
                                'src_tokens_id': sample['net_input']['src_tokens'][j],
                                'trg_tokens_id': sample['target'][j],
                                'cross_attn': cross_attn[j][:target_length[j],
                                              :source_length[j]].detach().cpu().numpy().astype(
                                    np.float16) if args.save_attn_weights else None,
                                'self_attn': self_attn[j][:target_length[j],
                                             :target_length[j]].detach().cpu().numpy().astype(
                                    np.float16) if args.save_attn_weights else None,
                            }
                        )

                        cur_token_idx = cur_token_idx + target_length[j].item()

                # save to the dstore
                current_batch_count = target.size(0)
                if dstore_idx + current_batch_count > args.dstore_size:
                    reduce_size = args.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
                    if args.save_plain_text:
                        src_tokens = src_tokens[:reduce_size, :]
                else:
                    reduce_size = current_batch_count

                if args.dstore_fp16:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float16)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)
                else:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float32)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)

                dstore_idx += reduce_size

                # print(dstore_idx)
                progress.log({'dstore_size': dstore_idx}, step=i)
                if dstore_idx > args.dstore_size:
                    print('much more than dstore size break')
                    break

                elif dstore_idx == args.dstore_size:
                    print('just fill')
            # -------- end, by

        print(dstore_idx)

    if args.save_plain_text:

        for sent_dict in tqdm(sentences_array):
            sent_dict['src_sent'] = src_dict.string(sent_dict['src_tokens_id'], return_list=False,
                                                    extra_symbols_to_ignore=[src_dict.pad()])
            sent_dict['trg_tokens'] = tgt_dict.string(sent_dict['trg_tokens_id'], return_list=True,
                                                      extra_symbols_to_ignore=[tgt_dict.pad()])
            sent_dict['trg_tokens'] = sent_dict['trg_tokens'].split(' ')
            sent_dict['trg_tokens'].append('<eos>')
            del sent_dict['src_tokens_id']
            del sent_dict['trg_tokens_id']

        print(sentences_array[1000])

        with open(args.dstore_mmap + '/text.dstore', 'wb') as f:
            import pickle
            pickle.dump({'locate_dict': locate_dict, 'sent_dict': sentences_array}, f)


def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
