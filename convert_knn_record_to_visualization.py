import pickle
import json
import numpy as np
import pathlib
import editdistance
import argparse


def get_context_using_idx(locate_dict, sentence_dict, idx, comparable_sent: list = None):
    if idx >= len(locate_dict):
        print(idx)

        return {'src': 'NULL', 'trg_tokens': 'NULL', 'cur_trg_idx': 'NULL'}

    sent_idx = locate_dict[idx]
    sent = sentence_dict[sent_idx]

    token_bias = idx - sent['start_idx']
    src = sent['src_sent']
    trg_tokens = sent['trg_tokens']
    cur_trg_idx = token_bias

    if comparable_sent is not None:
        edit_dist = editdistance.eval(src.replace('@@', '').split(), comparable_sent)
    else:
        edit_dist = None

    src_tokens = [token.replace('@@', '') for token in src.split(' ')]
    src_tokens.append('E')

    trg_tokens = [token.replace('@@', '').replace('<eos>', 'E') for token in trg_tokens]
    trg_tokens = ['B'] + trg_tokens

    return {
        'src_tokens': src_tokens,
        'trg_tokens': trg_tokens,
        'cur_trg_idx': cur_trg_idx,
        'edit_distance': edit_dist,
        'self_attention': sent['self_attn'][cur_trg_idx].tolist(),  # numpy.array to list
        'cross_attention': sent['cross_attn'][cur_trg_idx].tolist(),  # numpy.array to list
    }


parser = argparse.ArgumentParser()
parser.add_argument('--context-dstore-file', type=str, help='the datastore file which saved the plain text context information')
parser.add_argument('--knn-record-file', type=str, help='the index file which saved the knn index for each token in test set')
parser.add_argument('--reference-record-file', type=str, help='the reference record file')
parser.add_argument('--result-save-file', type=str, help='the file which save the record as json file, and used to visualize')
args = parser.parse_args()

K = 8
knn_record_file = args.knn_record_file
reference_knn_record_file = args.reference_record_file
context_dstore_file = args.context_dstore_file
json_result_save_dir = args.result_save_file

pathlib.Path(json_result_save_dir).mkdir(parents=True, exist_ok=True)

max_uncertainty = 0

with open(knn_record_file, "rb") as f1:
    with open(context_dstore_file, "rb") as f2:
        with open(reference_knn_record_file, "rb") as f3:

            dstore = pickle.load(f2)  # [dict: {'distance', 'index'}]
            knn_record = pickle.load(f1)
            reference_knn_record = pickle.load(f3)

            for i, sent_knn_record in enumerate(knn_record):  # for each sent
                print('process ', i)

                sent_knn_json_dict = {
                    'source': [token for token in sent_knn_record['source'].split(' ') + ['E']],
                    'reference': [token.replace('@@', '') for token in sent_knn_record['reference'].split(' ') + ['E']],
                    'hypothesis': [token.replace('@@', '') for token in
                                   sent_knn_record['generation'].split(' ') + ['E']],
                    'hypothesis_prefix': [token.replace('@@', '') for token in
                                          ['B'] + sent_knn_record['generation'].split(' ')],
                    'uncertainty': sent_knn_record['uncertainty'],
                    'attention': sent_knn_record['attention'] if 'attention' in sent_knn_record else None,
                    'self_attention': sent_knn_record[
                        'self_attention'] if 'self_attention' in sent_knn_record else None,
                    'top_n_candidate_idx': [pos.split(' ') for pos in
                                            sent_knn_record['top_n_candidate_idx']],
                    'top_n_candidate_prob': [[round(val, 5) for val in pos] for pos in
                                             sent_knn_record['top_n_candidate_prob']],
                    'top_n_knn_candidate_idx': [pos.split(' ') for pos in
                                                sent_knn_record['top_n_knn_candidate_idx']],
                    'top_n_knn_candidate_prob': [[round(val, 5) for val in pos] for pos in
                                                 sent_knn_record['top_n_knn_candidate_prob']],
                    'knn_record': [],
                    'reference_knn_record': [],
                }

                # record max uncertainty
                if np.max(sent_knn_record['uncertainty']) > max_uncertainty:
                    max_uncertainty = np.max(sent_knn_record['uncertainty'])

                for j in range(len(sent_knn_record['index'])):  # for each token

                    sent_knn_json_dict['knn_record'].append([])

                    for k in range(K):  # for each neighbor
                        sent_knn_json_dict['knn_record'][-1].append(
                            get_context_using_idx(dstore['locate_dict'], dstore['sent_dict'],
                                                  idx=sent_knn_record['index'][j][k],
                                                  comparable_sent=sent_knn_json_dict['source']))
                        sent_knn_json_dict['knn_record'][-1][-1]['distance'] = sent_knn_record['distance'][j][k]

                cur_reference_knn_record = reference_knn_record[sent_knn_record['sample_id']]
                for j in range(len(cur_reference_knn_record['knn_index'])):

                    sent_knn_json_dict['reference_knn_record'].append([])

                    for k in range(K):
                        sent_knn_json_dict['reference_knn_record'][-1].append(
                            get_context_using_idx(dstore['locate_dict'], dstore['sent_dict'],
                                                  idx=cur_reference_knn_record['knn_index'][j][k])
                        )
                        sent_knn_json_dict['reference_knn_record'][-1][-1]['distance'] = \
                            cur_reference_knn_record['knn_distance'][j][k]

                sent_json_path = "{}/{}.json".format(json_result_save_dir, i)
                with open(sent_json_path, 'w') as fp:
                    json.dump(sent_knn_json_dict, fp)

print(max_uncertainty)
