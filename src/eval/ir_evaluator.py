from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

import logging
from tqdm import tqdm
from typing import List, Tuple

from src.eval.fever.scorer import evidence_macro_precision, evidence_macro_recall

from src.utils.data_utils import load_pickle, read_jsonl, pred_format_pyserini_to_fever, get_multiHop_and_singleHop_data, get_file_dir
from src.utils.config import parser
from src.utils.args import prepare_logger

args = parser.parse_args()
logger = logging.getLogger()

def evidence_retrieval_score(predictions, max_evidence=5):
    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0
    
    for instance in predictions:
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'
        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec) if (pr + rec) != 0 else 0.0

    return pr, rec, f1

def get_docid_from_evi(evi):
    return evi[2]

def check_doc_level_hit(evidence_groups: List[List[Tuple]]=None, 
                          topk_doc_ids:set=None) -> bool:
    
    evidence_docs = (
                        [
                            {evidence[2] for evidence in evidence_set}
                            for evidence_set in evidence_groups
                        ]
                    )

    if any(
        evidence_doc.issubset(topk_doc_ids) for evidence_doc in evidence_docs
    ):
        return True
    else:
        return False

def eval_doc_recall_in_fever_format(retrievl_results: List[dict]=None, 
                                    max_evidence=5
                     ) -> float:
    
    hits = 0
    total = 0

    docid_nums = []
    
    for item in retrievl_results:
        if item['label'] == 'NOT ENOUGH INFO':
            continue
            
        total += 1
        
        pred_doc_ids = set([evi[0] for evi in item['predicted_evidence'][:max_evidence]])       
        docid_nums.append(len(pred_doc_ids))
            
        if check_doc_level_hit(evidence_groups=item['evidence'], 
                                  topk_doc_ids=pred_doc_ids):
            hits += 1

    logger.info(f"Number of verifiable claims: {total}")
    if docid_nums != []:
        logger.info(f"doc_recall: {hits/total}")
        logger.info(f"Max retrieved doids: {max(docid_nums)}")
        logger.info(f"Min retrieved doids: {min(docid_nums)}")
        logger.info(f"Average retrieved doids: {sum(docid_nums) / len(docid_nums)}")


def eval_for_each_claim(
                        hits,
                        hits_1,
                        possible_count,
                        possible_count_1,
                        all_count,
                        all_count_1,
                        topk_preds: List[int]=None,
                        top_doc_ids: set=None,
                        evidence_sets: List[set]=None,
                        evidence_docs: List[set]=None,
                        debug: bool=False,                        
                        ):
    """
    Checkes if there is a hit among the topk envidence.
    """

    # Evaluation Part
    is_one = min([len(evidence_set) for evidence_set in evidence_sets]) == 1 # Check if one-hop evidence.
    if is_one:
        all_count_1 += 1
    topk_lines = set([(line[0], str(line[1])) for line in topk_preds])

    if any(
        evidence_set.issubset(topk_lines) for evidence_set in evidence_sets
    ):
        hits += 1
        if is_one:
            hits_1 += 1
    if any(
        evidence_doc.issubset(top_doc_ids) for evidence_doc in evidence_docs
    ):
        possible_count += 1
        if is_one:
            possible_count_1 += 1
                
    all_count += 1

    return (
            hits,
            hits_1,
            possible_count,
            possible_count_1,
            all_count,
            all_count_1,
            )

def custom_eval(retrievl_results: List[dict]=None, 
                                    max_evidence=5
                     ) -> float:
    hits = 0
    hits_1 = 0
    possible_count = 0
    possible_count_1 = 0
    all_count = 0
    all_count_1 = 0

    for item in retrievl_results:
        if item['label'] == "NOT ENOUGH INFO":
            continue

        topk_preds = item['predicted_evidence'][:max_evidence]
        logger.debug(f"topk_preds: {topk_preds}")

        top_doc_ids = list(set([evi[0] for evi in topk_preds]))

        # A list of annoated evidence groups, each group has a set of (page_id, line_number) pairs.
        # Note, there could be duplicated evidence groups due to parallel annotation.
        evidence_sets = (
                        [
                            {
                                (evidence[2], str(evidence[3]))
                                for evidence in evidence_set
                            }
                            for evidence_set in item['evidence']
                        ]
                        if item['evidence']
                        else [set()]
                    )
        
        # A list of document-level annotated evidence, i.e., each evidence group has a set of 'page_id'.
        evidence_docs = (
                        [
                            {evidence[2] for evidence in evidence_set}
                            for evidence_set in item['evidence']
                        ]
                    )
        
        hits, hits_1, possible_count, possible_count_1, all_count, all_count_1 = eval_for_each_claim(
                                                                                                    hits,
                                                                                                    hits_1,
                                                                                                    possible_count,
                                                                                                    possible_count_1,
                                                                                                    all_count,
                                                                                                    all_count_1,
                                                                                                    topk_preds=topk_preds,
                                                                                                    top_doc_ids=top_doc_ids,
                                                                                                    evidence_sets=evidence_sets,                                                                                                                   evidence_sets=evidence_sets,
                                                                                                    evidence_docs=evidence_docs,                      
                                                                                                    )
    logger.info(
        "All examples sentence-level evidence hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits, all_count, hits / (all_count + 1e-6)
        )
    )
    logger.info(
        "All examples document-level evidence Can-hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits, possible_count, hits / (possible_count + 1e-6)
        )
    )
    logger.info(
        "One-hop sentence-level evidence hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits_1, all_count_1, hits_1 / (all_count_1 + 1e-6)
        )
    )
    logger.info(
        "One-hop document-level evidence Can-hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, hits_1, possible_count_1, hits_1 / (possible_count_1 + 1e-6)
        )
    )

    logger.info(
        "All examples document-level evidence hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, possible_count, all_count, possible_count / (all_count + 1e-6)
        )
    )

    logger.info(
        "One-hop document-level evidence hit ratio(fever-recall) @{}: {}/{} = {:.3f}".format(
            max_evidence, possible_count_1, all_count_1, possible_count_1 / (all_count_1 + 1e-6)
        )
    )


def main(args):
    max_evidence = [1, 5, 10, 20, 25, 50, 100, 200, 400, 500, 1000, 2048]
    logger.info(f"Loading result from: {args.retrieval_result_path}")
    try:
        result = load_pickle(args.retrieval_result_path)
    except:
        result = read_jsonl(args.retrieval_result_path)

    if args.debug:
        result = result[:10]
    logger.info("Done.")

    fever_result = pred_format_pyserini_to_fever(result, singleHopNumbers=args.singleHopNumbers)
    data_loose_single, data_strict_single, data_multi_strict_level0, data_multi_strict_level1, data_multi_strict_level2, data_multi_strict_level3 = get_multiHop_and_singleHop_data(fever_result)
    logger.info(f"len(fever_result): {len(fever_result)}")
    logger.info(f"len(data_loose_single): {len(data_loose_single)}")
    logger.info(f"len(data_strict_single): {len(data_strict_single)}")
    logger.info(f"len(data_multi_strict_level0): {len(data_multi_strict_level0)}")
    logger.info(f"len(data_multi_strict_level1): {len(data_multi_strict_level1)}")
    logger.info(f"len(data_multi_strict_level2): {len(data_multi_strict_level2)}")
    logger.info(f"len(data_multi_strict_level3): {len(data_multi_strict_level3)}")
    
    result_type = ['ALL', 'LooseSingleHop', 'StrictSingleHop', 'StrictMultiHop_Level_0', 'StrictMultiHop_Level_1', 'StrictMultiHop_Level_2', 'StrictMultiHop_Level_3']
    for idx, r in enumerate([fever_result, 
                             data_loose_single, 
                             data_strict_single, 
                             data_multi_strict_level0,
                             data_multi_strict_level1,
                             data_multi_strict_level2,
                             data_multi_strict_level3]):
        logger.info(f"\n\n====================================================")
        logger.info(f"================== {result_type[idx]} =================")
        logger.info(f"======================================================")
        for me in tqdm(max_evidence, desc='Evaluating topk retrieval results.'):
            pr, rec, f1 = evidence_retrieval_score(r, max_evidence=me)
            logger.info(f"\n******* retrieval scores @ {me}**********")
            logger.info(f"pr: {pr}")
            logger.info(f"rec: {rec}")
            logger.info(f"f1: {f1}")

            eval_doc_recall_in_fever_format(r, max_evidence=me)

            custom_eval(r, max_evidence=me)

if __name__ == '__main__':
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(get_file_dir(args.retrieval_result_path), "ir_eval.log"))
    main(args)
    