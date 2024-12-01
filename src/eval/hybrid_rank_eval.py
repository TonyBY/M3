import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import logging
from tqdm import tqdm
import copy
from typing import List

import transformers
transformers.logging.set_verbosity_error()

from src.eval.ir_evaluator import eval_doc_recall_in_fever_format, custom_eval
from src.eval.fever.scorer import fever_score
from src.utils.data_utils import make_directory, load_pickle, dump_pickle, get_file_name
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()
sep='|#SEP#|'


def get_hybrid_results_for_each_claim(
                                        dense_results: List[dict]=None, 
                                        sparse_results: List[dict]=None, 
                                        alpha: float=None, 
                                        k: int=None, 
                                        normalization: bool=False, 
                                        weight_on_dense: bool=False,
                                        ) -> List[dict]:
    
    dense_hits = {hit['id']: float(hit['score']) for hit in dense_results}
    sparse_hits = {hit['id']: float(hit['score']) for hit in sparse_results}
    
    hybrid_result = []
    min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
    max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
    min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
    max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
    
    logger.debug(f"min_dense_score: {min_dense_score}")
    logger.debug(f"max_dense_score: {max_dense_score}")
    logger.debug(f"min_dense_score: {min_dense_score}")
    logger.debug(f"min_sparse_score: {min_sparse_score}")
    logger.debug(f"max_sparse_score: {max_sparse_score}")
    
    for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
        if doc not in dense_hits:
            sparse_score = sparse_hits[doc]
            dense_score = min_dense_score
        elif doc not in sparse_hits:
            sparse_score = min_sparse_score
            dense_score = dense_hits[doc]
        else:
            sparse_score = sparse_hits[doc]
            dense_score = dense_hits[doc]
        if normalization:
            sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
                           / (max_sparse_score - min_sparse_score)
            dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
                          / (max_dense_score - min_dense_score)
        if sparse_score > dense_score:
            logger.debug(f"-------------")
            logger.debug(f"sparse_score: {sparse_score}")
            logger.debug(f"dense_score: {dense_score}")
        score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score
        hybrid_result.append({'id': doc, 'score': score})
    return sorted(hybrid_result, key=lambda x: float(x['score']), reverse=True)[:k]


def hybrid_results(
                    dense_results: List[dict]=None,
                    sparse_results: List[dict]=None,
                    alpha: float=0.1,
                    normalization: bool=False,
                    weight_on_dense: bool=False,
                    topk: int=2048) -> List[dict]:
    """
    params:
        alpha: # weight on the sparse score when weight_on_dense == False, otherwise, it's the weight on dense scores.
    """
    hybrid_data = []
    fever_data = []
    for dsr_item, bm25_item in tqdm(zip(dense_results, sparse_results), 
                                    total=len(dense_results),
                                    desc="Merging dense and sparse retriever's results...",
                                   ):
        if dsr_item['id'] != bm25_item['id']:
            print(dsr_item['id'], dsr_item['claim'])
            print(bm25_item['id'], bm25_item['claim'])
            raise Exception("Error, dense results and sparse results are not aligned.")
            break

        dense_hits = dsr_item['context']
        sparse_hits = bm25_item['context']
        hybrid_results = get_hybrid_results_for_each_claim(
                                                            dense_results=dense_hits, 
                                                            sparse_results=sparse_hits, 
                                                            alpha=alpha, 
                                                            k=topk, 
                                                            normalization=normalization, 
                                                            weight_on_dense=weight_on_dense,
                                                            )
        item = copy.deepcopy(dsr_item)
        item['context'] = hybrid_results
        hybrid_data.append(item)
        
        if 'evidence' in item:
            fever_item = {"id": item['id'],
                        "label": item['label'],
                        "evidence": item['evidence'],
                        "predicted_label": "",
                        "predicted_evidence": [[evi['id'].split('|#SEP#|')[0], int(evi['id'].split('|#SEP#|')[1])] for evi in hybrid_results],
                        }
        else:
            fever_item = {"id": item['id'],
                        "predicted_label": "",
                        "predicted_evidence": [[evi['id'].split('|#SEP#|')[0], int(evi['id'].split('|#SEP#|')[1])] for evi in hybrid_results],
                        }
        fever_data.append(fever_item)
    return hybrid_data, fever_data

def main(args):
    dense_results=load_pickle(args.dense_results_path)
    sparse_results=load_pickle(args.sparse_results_path)

    topk = 2048

    logger.info(f"args.sparse_results_path.lower(): {args.sparse_results_path.lower()}")
    if 'evidence' in dense_results[0] and ('train' not in args.sparse_results_path.lower()):
        logger.info("Start tuning hyperparameters...")
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        normalization = [False, True]
        weight_on_dense = [False, True]
        
        result_type="ALL"
        me=200

        best_recall = 0
        best_hybrid_results = None
        best_hybrid_results_in_fever_format = None
        best_alpha = None
        best_normalization = None
        best_weight_on_dense = None

        for norm in normalization:
            for w_on_d in weight_on_dense:
                for a in alpha:
                    logger.info("########################################################################")
                    logger.info("########################################################################")
                    logger.info(f"norm: {norm}")
                    logger.info(f"w_on_d: {w_on_d}")
                    logger.info(f"a: {a}")
                    hybrid_data, fever_data = hybrid_results(
                                                                dense_results=dense_results,
                                                                sparse_results=sparse_results,
                                                                alpha=a,
                                                                normalization=norm,
                                                                weight_on_dense=w_on_d,
                                                                topk=topk,
                                                            )
                    
                    r = fever_data
                    logger.info(f"\n\n====================================================")
                    logger.info(f"================== {result_type} =================")
                    logger.info(f"======================================================")
                    
                    strict_score, acc_score, pr, rec, f1 = fever_score(r,  max_evidence=me)
                    logger.info(f"\n******* fever scores @ {me}**********")
                    logger.info(f"strict_score: {strict_score}")
                    logger.info(f"acc_score: {acc_score}")
                    logger.info(f"pr: {pr}")
                    logger.info(f"rec: {rec}")
                    logger.info(f"f1: {f1}")

                    eval_doc_recall_in_fever_format(r, max_evidence=me)

                    custom_eval(r, max_evidence=me)

                    if rec > best_recall:
                        best_recall = rec
                        best_alpha = a
                        best_normalization = norm
                        best_weight_on_dense = w_on_d
                        best_hybrid_results = hybrid_data
                        best_hybrid_results_in_fever_format = fever_data
                        logger.info(f"best_recall@{me}: {best_recall}")
                        logger.info(f"best_alpha: {best_alpha}")
                        logger.info(f"best_normalization: {best_normalization}")
                        logger.info(f"best_weight_on_dense: {best_weight_on_dense}")
        
        logger.info("##############################################################")
        logger.info("Final conclusion:")
        logger.info(f"best_recall@{me}: {best_recall}")
        logger.info(f"best_alpha: {best_alpha}")
        logger.info(f"best_normalization: {best_normalization}")
        logger.info(f"best_weight_on_dense: {best_weight_on_dense}")

        output_path = os.path.join(args.hybric_search_dir, f'hybrid_RecAt{me}-{rec:.3f}_' + get_file_name(args.dense_results_path) + '.pkl')
        logger.info(f"Saving best_hybrid_results to: {output_path}")
        dump_pickle(best_hybrid_results, output_path)

        logger.info("##############################################################")
        for me in [1, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000, 2048, 4096]:
            logger.info(f"\n\n====================================================")
            logger.info(f"================== top-{me} =================")
            logger.info(f"======================================================")
            r = best_hybrid_results_in_fever_format
            strict_score, acc_score, pr, rec, f1 = fever_score(r,  max_evidence=me)

            logger.info(f"\n******* fever scores @ {me}**********")
            logger.info(f"strict_score: {strict_score}")
            logger.info(f"acc_score: {acc_score}")
            logger.info(f"pr: {pr}")
            logger.info(f"rec: {rec}")
            logger.info(f"f1: {f1}")

            eval_doc_recall_in_fever_format(r, max_evidence=me)

            custom_eval(r, max_evidence=me)
    else:
        logger.info("########################################################################")
        logger.info("########################################################################")
        logger.info(f"alpha: {args.alpha}")
        logger.info(f"normalization: {args.normalization}")
        logger.info(f"weight_on_dense: {args.weight_on_dense}")
        hybrid_data, fever_data = hybrid_results(
                                                    dense_results=dense_results,
                                                    sparse_results=sparse_results,
                                                    alpha=args.alpha,
                                                    normalization=args.normalization,
                                                    weight_on_dense=args.weight_on_dense,
                                                    topk=topk,
                                                )
        output_path = os.path.join(args.hybric_search_dir, f'hybrid_' + get_file_name(args.dense_results_path) + '.pkl')
        logger.info(f"Saving best_hybrid_results to: {output_path}")
        dump_pickle(hybrid_data, output_path)

if __name__ == '__main__':
    make_directory(args.hybric_search_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.hybric_search_dir, f"hybrid_eval.log"))
    logger.info(args)

    main(args)

    logger.info("============================================")
    logger.info("============================================")
    logger.info("ALL DONE.")
