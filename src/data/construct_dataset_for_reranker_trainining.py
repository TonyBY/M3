import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

import random
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from typing import List, Any
import math
import time
import logging
from math import ceil

from src.utils.data_utils import get_sentence_by_id, process_evid, split_list, merge_mp_results, load_pickle, dump_pickle, get_label_num, make_directory, get_file_name
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

"""
There are 1960 claims have multihop evidence annotations in the FEVER dev set.
There are 756 claims have both single and multihop evidence annotations in the FEVER dev set.
There are 1738 claims have legal multihop evidence annotations in the FEVER dev set.
There are 446 illegal multihop evidence annotations in the FEVER dev set
"""
def drop_duplicated_evidence_group(evidence_group_list: List[List[Any]]) -> List[List[Any]]:
    unique_evidence_group_idx = []
    viewed_evidence_group = set()
    for idx, eg in enumerate(evidence_group_list):
        evidence_group = ""
        for e in eg:
            sentence_id = f"{e[2]}_{e[3]}"
            evidence_group += sentence_id
        if evidence_group not in viewed_evidence_group:
            viewed_evidence_group.add(evidence_group)
            unique_evidence_group_idx.append(idx)
        else:
            continue
    return [evidence_group_list[i] for i in unique_evidence_group_idx]


def get_pos_neg_samples(first_hop_search_results,
                        wiki_line_dict: dict=None,
                        use_mnli_labels: bool=True, 
                        num_neg_samples: int=100, 
                        sep: str='|#SEP#|', 
                        RANDOM_SEED=random.Random(0),
                        worker_id: int=-1,
                        add_single_hop_egs: bool=True,
                        add_multi_hop_egs: bool=True,
                        NEGATIVE_SAMPLE_MULTIPLIER: int=3,
                        joint_reranking: bool=False,
                        singleHopNumbers: int=None,
                        multiHopNumbers: int=None,
                        ) -> List[list]:
    
    worker_id = 'Solo' if worker_id == -1 else worker_id
    
    multihop_evi_cunt = 0
    legal_multihop_evi_cunt = 0
    illegal_multihop_evi_cunt = 0
    clumped_dataset = []
    for item in tqdm(first_hop_search_results, 
                     desc=f">>Worker {worker_id}: Constructing datasets for sentence reranking..."):
        if item['label'] == 'NOT ENOUGH INFO':
            continue

        claim = item['claim']
        label = item['label']

        positive_label = get_label_num(label) if use_mnli_labels else 1
        negative_label = 1 if use_mnli_labels else 0

        evidence_groups = drop_duplicated_evidence_group(item['evidence'])

        multi_hop_egs = [eg for eg in evidence_groups if len(eg) > 1]
        single_hop_egs = [eg for eg in evidence_groups if len(eg) == 1]

        signle_hop_evi_set = set([sep.join([eg[0][2], str(eg[0][3])]) for eg in single_hop_egs])
        claim_evidence_set = signle_hop_evi_set

        if add_single_hop_egs:
            positives = [
                            [claim, process_evid(evi.split(sep)[0]), process_evid(get_sentence_by_id(evi, wiki_line_dict)), positive_label, 2] #[claim, evi_title, evi_content, nli_label, suficient_label]
                            for evi in signle_hop_evi_set
                        ]
        else:
            positives = []

        logger.debug(f"SingleHop positives == []: {positives == []}")

        # Add legal_multi_hop evidence.
        if add_multi_hop_egs and multi_hop_egs:
            multihop_evi_cunt += 1

            if signle_hop_evi_set and add_single_hop_egs:
                legal_multi_hop_egs = []
                for eg in multi_hop_egs:
                    LEGAL_MULTI = True
                    for evi in eg:
                        if (sep.join([evi[2], str(evi[3])])) in signle_hop_evi_set:
                            LEGAL_MULTI = False
                            illegal_multihop_evi_cunt += 1
                            break
                    if LEGAL_MULTI:
                        legal_multi_hop_egs.append(eg)
            else:
                legal_multi_hop_egs = multi_hop_egs

            legal_multihop_evi_set = set()
            for eg in legal_multi_hop_egs:
                for evi in eg:
                    legal_multihop_evi_set.add(sep.join([evi[2], str(evi[3])]))

            if legal_multihop_evi_set.intersection(signle_hop_evi_set) and signle_hop_evi_set and add_single_hop_egs:
                raise Exception("legal_multihop_evi_set cannot have intersection with signle_hop_evi_set.")
            
            claim_evidence_set.update(legal_multihop_evi_set)
            
            logger.debug(f"############### {legal_multihop_evi_cunt}, id: {item['id']} #####################")
            logger.debug("####################################")
            logger.debug(f"legal_multihop_evi_set: {legal_multihop_evi_set}")

            if legal_multihop_evi_set:
                legal_multihop_evi_cunt += 1

                # When also adding single_hop_egs, multi-hop reranking examples are naitive, i.e., [original_claim, any_evi_in_the_multihop_evi_group]
                if add_single_hop_egs: 
                    positives.extend( [
                                        [claim, process_evid(evi.split(sep)[0]), process_evid(get_sentence_by_id(evi, wiki_line_dict)), positive_label, 0]
                                            for evi in legal_multihop_evi_set
                                        ]
                                    )
                # When not adding single_hop_egs, we assume using the DSR-M's result to construct multi-hop examples, which are iterative reranking examples, 
                # i.e., [claim + an_evi_in_the_multihop_evi_group, another_evi_in_the_same_evi_group]
                else: 
                    claim_pairs = []
                    for eg in legal_multi_hop_egs:
                        if len(eg) == 2:
                            suf_label = 2 # sufficient
                        else:
                            suf_label = 0 # insufficient

                        evi_1 = sep.join([eg[0][2], str(eg[0][3])])
                        evi_title_1 = process_evid(eg[0][2])
                        evi_content_1 = process_evid(get_sentence_by_id(evi_1, wiki_line_dict))
                        claim_multi_1 = ' -- '.join([claim, evi_title_1 + ' . ' + evi_content_1])
                       
                        evi_2 = sep.join([eg[1][2], str(eg[1][3])])
                        evi_title_2 = process_evid(eg[1][2])
                        evi_content_2 = process_evid(get_sentence_by_id(evi_2, wiki_line_dict))
                        claim_multi_2 = ' -- '.join([claim, evi_title_2 + ' . ' + evi_content_2])

                        logger.debug(f"claim_multi_1: {claim_multi_1}")
                        logger.debug(f"claim_multi_2: {claim_multi_2}")

                        claim_pairs.append((claim_multi_1, claim_multi_2))

                        positives.extend([
                                            [claim_multi_1, evi_title_2, evi_content_2, positive_label, suf_label],
                                            [claim_multi_2, evi_title_1, evi_content_1, positive_label, suf_label]
                                            ]
                                )
                        logger.debug(f"claim_pairs: {claim_pairs}")
                logger.debug(f"len(positives) after added multi-hop positives numbers: {len(positives)}")
                
                        
        if positives == []:
            continue

        if positive_label == 0:  # over sample refutes (when use_mnli_labels)
            positives = positives * NEGATIVE_SAMPLE_MULTIPLIER

        if add_single_hop_egs: # We can use both single and multi-hop examples to train SRR-S, but we can only use multi-hop examples to train SRR-M.
            if joint_reranking:
                logger.debug(f"Sampling negtives from: item['context'][:{singleHopNumbers}] and item['multihop_context'][:{multiHopNumbers}]...")
                logger.debug(f"len(item['context']): {len(item['context'])}")
                logger.debug(f"len(item['multihop_context']): {len(item['multihop_context'])}")
                negatives_results = item['context'][:singleHopNumbers] + item['multihop_context'][:multiHopNumbers]
                logger.debug(f"len(negatives_results): {len(negatives_results)}")
            else:
                logger.debug(f"Sampling negtives from: item['context']...")
                negatives_results = item['context']
        else:
            logger.debug(f"Sampling negtives from: item['multihop_context']...")
            negatives_results = item['multihop_context']

        filtered_negatives = set()
        for result in negatives_results:
            title = process_evid(result['id'].split(sep)[0])
            score = result['score']
            sent_text =  process_evid(get_sentence_by_id(result['id'], wiki_line_dict))
                
            if  result['id'] not in claim_evidence_set:
                filtered_negatives.add((title, 
                                        sent_text,
                                        score)
                                      )

        filtered_negatives = list(filtered_negatives)

        choices = filtered_negatives

        logger.debug(f"len(choices): {len(choices)}")

        unique_choices = set()
        for choice in choices:
            if choice not in unique_choices:
                unique_choices.add(choice)
            if len(unique_choices) == num_neg_samples:
                break
        logger.debug(f"len(unique_choices): {len(unique_choices)}")

        if add_single_hop_egs:
            negatives = [
                [claim, sample[0], sample[1], negative_label, 1] for sample in unique_choices
            ]
        else:
            negatives = []
            for pair in claim_pairs:
                negatives.extend([
                                    [pair[0], sample[0], sample[1], negative_label, 1] for sample in list(unique_choices)[:max(10, ceil(num_neg_samples/(2 * len(claim_pairs))))]
                                ] + [
                                    [pair[1], sample[0], sample[1], negative_label, 1] for sample in list(unique_choices)[:max(10, ceil(num_neg_samples/(2 * len(claim_pairs))))]
                                ])
        logger.debug(f"len(negatives): {len(negatives)}")
        logger.debug(f"negatives: {negatives}")

        clumped_dataset.extend(positives + negatives)

    return clumped_dataset

def get_balanced_dev_data(dsr_search_result_dev: pd.DataFrame, use_mnli_labels: bool) -> pd.DataFrame:
    logger.info("Start balanced nli dve set...")
    if use_mnli_labels:
        num_sup = len(dsr_search_result_dev.iloc[dsr_search_result_dev.index[dsr_search_result_dev.nli_label==2]])
        num_ref = len(dsr_search_result_dev.iloc[dsr_search_result_dev.index[dsr_search_result_dev.nli_label==0]])
        num_nei = len(dsr_search_result_dev.iloc[dsr_search_result_dev.index[dsr_search_result_dev.nli_label==1]])
        n = min(num_sup, num_ref, num_nei)
        balanced_dsr_search_result_dev = dsr_search_result_dev.groupby("nli_label").sample(n=n, random_state=1).reset_index(drop = True)

        logger.info(f"num_sup: {num_sup}")
        logger.info(f"num_ref: {num_ref}")
        logger.info(f"num_nei: {num_nei}")
        logger.info(f"min(num_sup, num_ref, num_nei): {n}")
    else:
        num_pos = len(dsr_search_result_dev.iloc[dsr_search_result_dev.index[dsr_search_result_dev.nli_label==1]])
        num_neg = len(dsr_search_result_dev.iloc[dsr_search_result_dev.index[dsr_search_result_dev.nli_label==0]])
        n = min(num_pos, num_neg)
        balanced_dsr_search_result_dev = dsr_search_result_dev.groupby("nli_label").sample(n=n, random_state=1).reset_index(drop = True)

        logger.info(f"num_pos: {num_pos}")
        logger.info(f"num_neg: {num_neg}")
        logger.info(f"min(num_pos, num_neg): {n}")
    
    return balanced_dsr_search_result_dev

def main(args):
    start = time.time()

    logger.info(f"Loading first_hop_search_results from: {args.first_hop_search_results_path}")
    first_hop_search_results = load_pickle(args.first_hop_search_results_path)
    if args.debug:
        if not args.add_single_hop_egs:
            first_hop_search_results = [item for item in tqdm(first_hop_search_results) if any(len(eg) > 1 for eg in item['evidence'])]
            logger.debug(f"len(first_hop_search_results): {len(first_hop_search_results)}")
        first_hop_search_results = first_hop_search_results[:100]

    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    file_size = len(first_hop_search_results)
    num_workers = min(mp.cpu_count(), args.max_num_process)
    num_workers = num_workers if file_size // num_workers > 1 else 1

    logger.info(f"num_workers: {num_workers}")

    if num_workers > 1:
        chunk_size = math.ceil(file_size / num_workers)
        chunks = split_list(first_hop_search_results, chunk_num=num_workers)

        logger.info(f"file_size: {file_size}")
        logger.info(f"chunk_size: {chunk_size}")
        logger.info(f"number of chunks: {len(chunks)}")

        results = []
        pool = mp.Pool(num_workers)
        for i in range(len(chunks)):
            logger.info(f"starting worker: {i}")
            proc = pool.apply_async(
                                    get_pos_neg_samples, 
                                        (chunks[i],), 
                                        dict(
                                                wiki_line_dict = wiki_line_dict,
                                                num_neg_samples = args.num_neg_samples,
                                                use_mnli_labels = args.use_mnli_labels,
                                                worker_id = i,
                                                add_single_hop_egs = args.add_single_hop_egs,
                                                add_multi_hop_egs = args.add_multi_hop_egs,
                                                joint_reranking = args.joint_reranking,
                                                singleHopNumbers = args.singleHopNumbers,
                                                multiHopNumbers = args.multiHopNumbers,
                                            )
                                    )
            results.append(proc)

        pool.close()
        pool.join()
        
        sentence_selection_dataset = merge_mp_results(results)
        
    else:
        sentence_selection_dataset = get_pos_neg_samples(
                                                            first_hop_search_results,
                                                            wiki_line_dict = wiki_line_dict,
                                                            num_neg_samples = args.num_neg_samples,
                                                            use_mnli_labels = args.use_mnli_labels,
                                                            add_single_hop_egs = args.add_single_hop_egs,
                                                            add_multi_hop_egs = args.add_multi_hop_egs,
                                                            joint_reranking = args.joint_reranking,
                                                            singleHopNumbers = args.singleHopNumbers,
                                                            multiHopNumbers = args.multiHopNumbers,
                                                        )

    end = time.time()
    total_time = end - start
    logger.info(f"total_time: {total_time}")

    logger.debug(f"len(sentence_selection_dataset): len(sentence_selection_dataset)")
    df = pd.DataFrame(
                sentence_selection_dataset,
                columns=["claim", "candidate_title", "candidate_sentence", "nli_label", "sufficiency_label"],
            )
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    logger.debug(f"df[:10]: {df[:10]}")
    if 'dev' in get_file_name(args.first_hop_search_results_path).lower():
        output_file_name = 'dev_nli_imbalanced'

        balanced_dsr_search_result_dev = get_balanced_dev_data(df, args.use_mnli_labels)
        dev_output_path = os.path.join(args.reranking_dir, "balanced_nli_dev.pkl")
        logger.info(f"Saving balanced nli dev set to: {dev_output_path}")
        dump_pickle(balanced_dsr_search_result_dev, dev_output_path)
    elif 'train' in get_file_name(args.first_hop_search_results_path).lower():
        output_file_name = 'train_nli'
    else:
        output_file_name = get_file_name(args.first_hop_search_results_path)


    dump_pickle(df, os.path.join(args.reranking_dir, output_file_name + '.pkl'))

if __name__ == "__main__":
    if args.debug:
        args.reranking_dir = args.reranking_dir + '_DEBUG'
    make_directory(args.reranking_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.reranking_dir, 'sentence_ranking_' + get_file_name(args.first_hop_search_results_path) + '.log'))
    logger.info(args)
    if args.add_single_hop_egs or args.add_multi_hop_egs:
        main(args)
        logger.info("All Done.")
    else:
        logger.info("Both args.add_single_hop_egs and args.add_multi_hop_egs are: False. Doing nothing. Exiting.")
        exit()
