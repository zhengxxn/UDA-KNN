import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--first-dstore', type=str)
parser.add_argument('--second-dstore', type=str)
parser.add_argument('--third-dstore', type=str)
parser.add_argument('--forth-dstore', type=str)
parser.add_argument('--dstore-size', type=int)
parser.add_argument('--dimension', type=int)
parser.add_argument('--export-path', type=str)

args = parser.parse_args()

first_keys = np.memmap(args.first_dstore + '/ordered_keys.npy', dtype=np.float16, mode='r',
                       shape=(args.dstore_size, args.dimension))
first_vals = np.memmap(args.first_dstore + '/ordered_vals.npy', dtype=np.int, mode='r',
                       shape=(args.dstore_size, 1))

first_keys = np.array(first_keys).astype(np.float32)
first_vals = np.array(first_vals)

second_keys = np.memmap(args.second_dstore + '/ordered_keys.npy', dtype=np.float16, mode='r',
                        shape=(args.dstore_size, args.dimension))
second_vals = np.memmap(args.second_dstore + '/ordered_vals.npy', dtype=np.int, mode='r',
                        shape=(args.dstore_size, 1))

second_keys = np.array(second_keys).astype(np.float32)
second_vals = np.array(second_vals)

third_keys = np.memmap(args.third_dstore + '/ordered_keys.npy', dtype=np.float16, mode='r',
                       shape=(args.dstore_size, args.dimension))
third_vals = np.memmap(args.third_dstore + '/ordered_vals.npy', dtype=np.int, mode='r',
                       shape=(args.dstore_size, 1))

third_keys = np.array(third_keys).astype(np.float32)
third_vals = np.array(third_vals)

forth_keys = np.memmap(args.forth_dstore + '/ordered_keys.npy', dtype=np.float16, mode='r',
                       shape=(args.dstore_size, args.dimension))
forth_vals = np.memmap(args.forth_dstore + '/ordered_vals.npy', dtype=np.int, mode='r',
                       shape=(args.dstore_size, 1))

forth_keys = np.array(forth_keys).astype(np.float32)
forth_vals = np.array(forth_vals)

# token_idx = 21956  # trigger
# token_idx = 1529  # google
# token_idx = 16029  # hosts
# token_idx = 2866  # network 712
# token_idx = 1681  # software
# token_idx = 7730  # profile
# token_idx = 8131  # channel
token_idx = 9896  # library


value_idx = np.where(first_vals == token_idx)[0]
# value_idx_of_second = np.where(second_vals == token_idx)[0]
# value_idx_of_third = np.where(third_vals == token_idx)[0]
# value_idx_of_forth = np.where(forth_vals == token_idx)[0]

# print(np.shape(value_idx_of_first))
# print(np.shape(value_idx_of_second))
# print(np.shape(value_idx_of_third))
print(np.shape(value_idx))

value_dict = {'first': first_keys[value_idx], 'second': second_keys[value_idx],
              'third': third_keys[value_idx], 'forth': forth_keys[value_idx]}

import pickle

with open(args.export_path, 'wb') as f:
    pickle.dump(value_dict, f)
