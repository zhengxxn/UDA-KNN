import argparse
import math
import torch
from tqdm import tqdm


def define_mask_span_distribution(_poisson_lambda):
    lambda_to_the_k = 1
    e_to_the_minus_lambda = math.exp(-_poisson_lambda)
    k_factorial = 1
    ps = []
    for k in range(0, 128):
        ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
        lambda_to_the_k *= _poisson_lambda
        k_factorial *= k + 1
        if ps[-1] < 0.0000001:
            break
    ps = torch.FloatTensor(ps)
    _mask_span_distribution = torch.distributions.Categorical(ps)

    return _mask_span_distribution


def get_word_starts(_tokens, _mask_whole_word=None):

    is_word_start = torch.zeros(len(_tokens))
    is_word_start[0] = 1

    if _mask_whole_word is not None:
        for i in range(1, len(_tokens)):
            if not _tokens[i - 1].endwith('@@'):
                is_word_start[i] = 1
    else:
        # we regard all tokens are word start, which we only use this case in our experiments
        is_word_start = torch.ones(len(_tokens))

    # is_word_start[0] = 0

    # set <eos> is not word start
    is_word_start[-1] = 0
    return is_word_start


def add_insertion_noise(_tokens, _p):

    if _p == 0.0:
        return _tokens

    num_tokens = len(_tokens)
    n = int(math.ceil(num_tokens * _p))

    # note: our final sent have (num_tokens + n) tokens,
    noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
    # note: in (num_tokens + n), we did not allow the last to be <mask>, as it should be <eos>
    noise_indices = torch.randperm(num_tokens + n - 1)[:n]

    noise_mask[noise_indices] = 1

    result = []
    idx_of_cur_tokens = 0

    for i in range(0, num_tokens + n):

        if noise_mask[i] == 1:
            result.append('<mask>')

        else:

            result.append(_tokens[idx_of_cur_tokens])
            idx_of_cur_tokens += 1

    return result


def add_whole_word_mask(_tokens, _p, _mask_span_distribution=None, _mask_random_ratio=0.0, _replace_length=0):
    is_word_start = get_word_starts(_tokens)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * _p))
    num_inserts = 0
    if num_to_mask == 0:
        return _tokens

    if _mask_span_distribution is not None:

        lengths = _mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        # accumulate the length
        cum_length = torch.cumsum(lengths, 0)

        # note: if the sum of length still small than num_to_mask, append new sample length ( why num_to_mask, not 1? )
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    _mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        # note: cum_length[i] >= num_to_mask here

        # note: cut off to make cum_length[i] == num_to_mask
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])

        # note: here num_to_mask is set to count of (mask span)
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts

        # note: i have no idea why we should add insertion noise here, so we ignore it
        if num_to_mask == 0:

            return add_insertion_noise(_tokens, num_inserts / len(_tokens))

        assert (lengths > 0).all()

    else:
        # each mask is 1 token
        lengths = torch.ones((num_to_mask,)).long()

    # note: this is for eos? i have no idea, eos should not be mask ?
    assert is_word_start[-1] == 0

    word_starts = is_word_start.nonzero(as_tuple=False)  # return the indices [num_of_non_zero, 1]
    # note: if we perform token mask, this is used to randomly select (num_to_mask) tokens
    # if we perform span mask, this is used to randomly select (num_to_mask) spans (which may cause overlap)
    indices = word_starts[
        torch.randperm(word_starts.size(0))[:num_to_mask]
    ].squeeze(1)

    # note: the index which replace by random token (which never occurred in plain text as we always set it to zero),
    # but not <mask>
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < _mask_random_ratio

    source_length = len(_tokens)
    # note: also to avoid eos to mask, we should remove from here
    assert source_length - 1 not in indices

    to_keep = torch.ones(source_length, dtype=torch.bool)

    # TODO: we assume this is for <eos>, so we remove it
    is_word_start[
        -1
    ] = 255  # acts as a long length, so spans don't go over the end of doc

    if _replace_length == 0:
        # replace_length means remove the token here, so to_keep set to 0
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        # source[indices] = self.mask_idx
        # _tokens[indices.tolist()] = '<mask>'
        for index in indices:
            _tokens[index] = '<mask>'
        # source[indices[mask_random]] = torch.randint(
        #     1, len(self.vocab), size=(mask_random.sum(),)
        # )

    if _mask_span_distribution is not None:
        # note: for each sentence, i think
        assert len(lengths.size()) == 1
        #
        assert lengths.size() == indices.size()

        # note: since we have mask the first token for each span
        lengths -= 1

        while indices.size(0) > 0:

            assert lengths.size() == indices.size()

            # note: we set is_word_start for all 1. This code means we want to remove the second token of each span
            lengths -= is_word_start[indices + 1].long()

            # a list of bool for each span,
            uncompleted = lengths >= 0

            # note: if is uncompleted, then we move to next idx for each span
            indices = indices[uncompleted] + 1

            # todo: have no idea here
            mask_random = mask_random[uncompleted]

            # remove the completed
            lengths = lengths[uncompleted]

            if _replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # = -1 means replace with same length mask
                # keep index, but replace it with [MASK]
                # _tokens[indices.tolist()] = '<mask>'
                for index in indices:
                    _tokens[index] = '<mask>'
                # source[indices[mask_random]] = torch.randint(
                #     1, len(self.vocab), size=(mask_random.sum(),)
                # )
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if _replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                # _tokens[indices.tolist()] = '<mask>'
                for index in indices:
                    _tokens[index] = '<mask>'
                # _tokens[indices[mask_random]] = torch.randint(
                #     1, len(self.vocab), size=(mask_random.sum(),)
                # )

            assert source_length - 1 not in indices

    _tokens = [token for i, token in enumerate(_tokens) if to_keep[i]]
    # _tokens = _tokens[to_keep]

    if num_inserts > 0:
        _tokens = add_insertion_noise(_tokens, num_inserts / len(_tokens))

    return _tokens


parser = argparse.ArgumentParser()
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

parser.add_argument("--input-file", default=None, type=str, help="make sure that the input is bpe tokenized")
parser.add_argument("--output-file", default=None, type=str)

args = parser.parse_args()

mask_ratio = args.denoising_mask_ratio
mask_length = args.denoising_mask_length
poisson_lambda = args.denoising_poisson_lambda
mask_span_distribution = None
if mask_length == "span-poisson":
    mask_span_distribution = define_mask_span_distribution(poisson_lambda)

mask_sents = []

with open(args.input_file, 'r') as f:

    sents = f.readlines()

    for sent in tqdm(sents):
        tokens = sent.strip('\n').split(' ')
        tokens.append('<eos>')
        length = len(tokens)

        if mask_ratio > 0:
            tokens = add_whole_word_mask(tokens, _p=mask_ratio, _mask_span_distribution=mask_span_distribution,
                                         _replace_length=1)

        del tokens[-1]
        new_sent = " ".join(tokens)
        mask_sents.append(new_sent)

with open(args.output_file, 'w') as f:
    f.write("\n".join(mask_sents))
    f.write('\n')
