import pickle
import argparse


def get_context_using_idx(locate_dict, sentence_dict, idx):

    sent_idx = locate_dict[idx]
    sent = sentence_dict[sent_idx]

    token_bias = idx - sent['start_idx']
    src = sent['src_sent']
    trg_prefix = " ".join(sent['trg_tokens'][:token_bias])
    trg_token = sent['trg_tokens'][token_bias]
    print("src: {}".format(src))
    print("trg_prefix: {}".format(trg_prefix))
    print("trg_token: {}".format(trg_token))


parser = argparse.ArgumentParser()
parser.add_argument('--context-dstore-file', type=str, help='the datastore file which saved the plain text context information')
parser.add_argument('--knn-record-file', type=str, help='the index file which saved the knn index for each token in test set')
args = parser.parse_args()


with open(args.knn_record_file, "rb") as f1:
    with open(args.context_dstore_file, "rb") as f2:

        dstore = pickle.load(f2)
        knn_record = pickle.load(f1)  # [dict: {'distance', 'index'}]

        print('src ', knn_record[1024]['source'])
        print('reference ', knn_record[1024]['reference'])
        print('generation ', knn_record[1024]['generation'])

        # for example
        for i in range(8):

            get_context_using_idx(dstore['locate_dict'], dstore['sent_dict'], idx=knn_record[1024]['index'][0][i])
            print(knn_record[1024]['distance'][0][i])
            print()


