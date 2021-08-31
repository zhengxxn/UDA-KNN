import numpy as np
import argparse


def cosine_similarity(a, b):
    # a [B, H]
    # b [B, H]
    print(np.shape(a))
    print(np.shape(b))
    dot_product = (a * b).sum(axis=1)  # [B]
    norm_a = np.linalg.norm(a, axis=1)  # [B]
    norm_b = np.linalg.norm(b, axis=1)  # [B]
    cos_sim = dot_product / (norm_a * norm_b)
    print(np.mean(cos_sim))


def l2_distances(a, b):
    dist = np.linalg.norm(a - b, axis=-1)
    mean_dist = np.mean(dist)
    print(mean_dist)


parser = argparse.ArgumentParser()
parser.add_argument('--first-dstore', type=str)
parser.add_argument('--second-dstore', type=str)
parser.add_argument('--dstore-size', type=int)
parser.add_argument('--dimension', type=int)

args = parser.parse_args()

print("load first store")
first_dstore = np.memmap(args.first_dstore, dtype=np.float16, mode='r',
                         shape=(args.dstore_size, args.dimension))
print("load second store")
second_dstore = np.memmap(args.second_dstore, dtype=np.float16, mode='r',
                          shape=(args.dstore_size, args.dimension))
print("load done")

if args.dstore_size > 3000000:
    random_choice = np.random.choice(np.arange(args.dstore_size), size=[3000000], replace=False)
    print("random sample done.")
    _first_dstore = np.array(first_dstore[random_choice])
    print("first sample done")
    _second_dstore = np.array(second_dstore[random_choice])
    print("second sample done")

else:
    _first_dstore = np.array(first_dstore)
    _second_dstore = np.array(second_dstore)

cosine_similarity(_first_dstore, _second_dstore)
l2_distances(_first_dstore, _second_dstore)
