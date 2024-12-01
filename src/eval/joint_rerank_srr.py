import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import make_directory, load_pickle, dump_pickle, \
                                 get_file_name

from typing import List
from tqdm import tqdm
import logging
import copy

from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

def get_hybrid_results_for_each_claim(  
                                        dense_results: List[dict]=None, 
                                        sparse_results: List[dict]=None, 
                                        alpha: float=None, 
                                        normalization: bool=False, 
                                        weight_on_dense: bool=False,
                                        metric: str='path',
                                        sep: str='|#SEP#|',
                                    ) -> List[dict]:
        
    try:
        dense_hits = {hit[0] + sep + str(hit[1]): float(hit[2]) for hit in dense_results}
    except:
        dense_hits = {hit['id']: float(hit['score']) for hit in dense_results}
   
    sparse_hits = {}
    for evi in sparse_results:
        sentID1 = evi[0]
        sentID2 = evi[2]
        sentScore1 = float(evi[1])
        sentScore2 = float(evi[3])
        if metric == "path":
            if sentID1 not in sparse_hits or sentScore2 > sparse_hits[sentID1]:
                sparse_hits[sentID1] = sentScore2
            if sentID2 not in sparse_hits or sentScore2 > sparse_hits[sentID2]:
                sparse_hits[sentID2] = sentScore2
        elif metric == 'sum':
            score = sentScore1 + sentScore2
            if sentID1 not in sparse_hits or score > sparse_hits[sentID1]:
                sparse_hits[sentID1] = score
            if sentID2 not in sparse_hits or score > sparse_hits[sentID2]:
                sparse_hits[sentID2] = score
        elif metric == 'product':
            score = sentScore1 * sentScore2
            if sentID1 not in sparse_hits or score > sparse_hits[sentID1]:
                sparse_hits[sentID1] = score
            if sentID2 not in sparse_hits or score > sparse_hits[sentID2]:
                sparse_hits[sentID2] = score
        else:
            raise Exception(f"Unknown meric: {metric}")  
    
    hybrid_result = []
    min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
    max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
    min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
    max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1

    # Note: DO NOT use:
    #   """ for key in set(sparse_hits.keys()) | set(dense_hits.keys()):""""
    #   Because set operation will change the sequence of keys in the dictionary in every new exeripent!
    #   This will cause issue in the sorted() function when multiple items have the same value.
    #   In this case, the sorted sequence will depend on the input sequence.
    doc_ids = []
    viewed_doc_ids = set()
    for key in list(sparse_hits.keys()) + list(dense_hits.keys()):
        if key not in viewed_doc_ids:
            viewed_doc_ids.add(key)
            doc_ids.append(key)
        else:
            continue

    logger.debug(f"doc_ids: {doc_ids}")

    for doc in doc_ids:
        sparse_score = sparse_hits[doc] if doc in sparse_hits else min_sparse_score
        dense_score = dense_hits[doc] if doc in dense_hits else min_dense_score
        if normalization:
            sparse_score = (sparse_score - min_sparse_score) \
                           / (max_sparse_score - min_sparse_score + 1e-10)
            dense_score = (dense_score - min_dense_score) \
                          / (max_dense_score - min_dense_score + 1e-10)
            
        score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score
        hybrid_result.append({'id': doc, 'score': score})

    return sorted(hybrid_result, key=lambda x: float(x['score']), reverse=True)

def get_valid_multihop_evi(multihop_reranked_context: List[list]=None,
                           metric: str=None,
                           th: float=None,) -> List[list]:
    valid_multihop_evi = []
    empty_multihop_ctx_cnt = 0
    for evi in multihop_reranked_context:
        if metric == 'path':
            score = float(evi[3])
        elif metric == 'sum':
            score = float(evi[1]) + float(evi[3])
        elif metric == 'product':
            score = float(evi[1]) * float(evi[3])
        else:
            raise Exception(f"Unknown metric: {metric}")

        if score > th:
            valid_multihop_evi.append(evi)
    if len(valid_multihop_evi) == 0:
        empty_multihop_ctx_cnt += 1
    return valid_multihop_evi

def eval_for_each_claim(
                        hits,
                        hits_1,
                        possible_count,
                        possible_count_1,
                        all_count,
                        all_count_1,
                        topk_preds: List[list]=None,
                        top_doc_ids: set=None,
                        evidence_sets: List[set]=None,
                        evidence_docs: List[set]=None,                      
                        sep: str='|#SEP#|'):
    """
    Checkes if there is a hit among the topk envidence.
    """
    # Re-ranked topk_evidence: a list of (page_id, line_id, relevance_score) tuplets.
    topk_evidence = set()
    for pred in topk_preds:
        doc_id, line_num = pred['id'].split(sep)
        topk_evidence.add((doc_id, line_num, float(pred['score'])))
    
    # Evaluation Part
    is_one = min([len(evidence_set) for evidence_set in evidence_sets]) == 1 # Check if one-hop evidence.
    if is_one:
        all_count_1 += 1
        
    topk_lines = set([tuple(prediction['id'].split(sep)) for prediction in topk_preds])
    
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

def merge_reranked_evi(items: List[dict]=None, 
                        alpha: float=None, 
                        k: int=5, 
                        normalization: bool=False, 
                        weight_on_dense: bool=False,
                        metric: str='path',
                        mhth: float=0.99,
                        sep: str='|#SEP#|',
                        singleHopNumbers: int=5,
                        naive_merge_discount_factor: float = None,):
    logger.info(f"type(alpha): {type(alpha)}")
    logger.info(f"alpha: {alpha}")
    logger.info(f"type(k): {type(k)}")
    logger.info(f"k: {k}")
    logger.info(f"type(normalization): {type(normalization)}")
    logger.info(f"normalization: {normalization}")
    logger.info(f"type(weight_on_dense): {type(weight_on_dense)}")
    logger.info(f"weight_on_dense: {weight_on_dense}")
    logger.info(f"type(metric): {type(metric)}")
    logger.info(f"metric: {metric}")
    logger.info(f"type(mhth): {type(mhth)}")
    logger.info(f"mhth: {mhth}")
    logger.info(f"type(singleHopNumbers): {type(singleHopNumbers)}")
    logger.info(f"singleHopNumbers: {singleHopNumbers}")
    logger.info(f"type(naive_merge_discount_factor): {type(naive_merge_discount_factor)}")
    logger.info(f"naive_merge_discount_factor: {naive_merge_discount_factor}")
    
    if  'reranked_context' not in items[0]:
        logger.info(f"singleHopNumbers: {singleHopNumbers}")

    merged_reranked_results = []
    for i in tqdm(range(len(items)), desc='Merging singlehop and multihop retrieval results...'):
        if  'reranked_context' in items[i]:
            first_hop_reranked_context = items[i]['reranked_context']
        else:
            first_hop_reranked_context = items[i]['context'][:singleHopNumbers]
        
        multihop_reranked_context = items[i]['multihop_reranked_context'] if 'multihop_reranked_context' in items[i] else []

        if multihop_reranked_context == []:
            raise Exception("No multihop_reranked_context is availabel in the data.")

        if naive_merge_discount_factor != None:
            if 'id' in multihop_reranked_context[0]:
                multihop_reranked_context = [{'id': ctx['id'], 'score': float(ctx['score']) * naive_merge_discount_factor} for ctx in multihop_reranked_context]
            else:
                multihop_reranked_context = [{'id': ctx[2], 'score': ctx[3] * naive_merge_discount_factor} for ctx in multihop_reranked_context]
            merged_evi = first_hop_reranked_context + multihop_reranked_context
            merged_evi = sorted(merged_evi, key=lambda x: float(x['score']), reverse=True)
        else:          
            valid_multihop_evi = get_valid_multihop_evi(multihop_reranked_context=multihop_reranked_context,
                                                        metric=metric,
                                                        th=mhth,
                                                        )
            merged_evi = get_hybrid_results_for_each_claim(
                                                            dense_results=first_hop_reranked_context, 
                                                            sparse_results=valid_multihop_evi, 
                                                            alpha=alpha, 
                                                            normalization=normalization, 
                                                            weight_on_dense=weight_on_dense,
                                                            metric=metric,
                                                            )
        
        items[i]['merged_retrieval'] = merged_evi[:k]
        if 'multihop_context' in items[i]:
            items[i].pop('multihop_context')

        merged_reranked_results.append(items[i])

    return merged_reranked_results

def get_docids_from_a_list_of_preds(preds:List[dict]=None, sep: str='|#SEP#|') -> List[str]:
    viewed_doc_ids = set()
    for pred in preds:
        doc_id = pred['id'].split(sep)[0]
        viewed_doc_ids.add(doc_id)
    return list(viewed_doc_ids)

def retrieval_evaluator(items: List[dict]=None, 
                        sep: str='|#SEP#|',
                       ):
    hits = 0
    hits_1 = 0
    possible_count = 0
    possible_count_1 = 0
    all_count = 0
    all_count_1 = 0
    
    for i in tqdm(range(len(items)), desc="Evaluating merged results..."):
        if items[i]['label'] == 'NOT ENOUGH INFO':
            continue
            
        topk_preds = items[i]['merged_retrieval'][:5]
        logger.debug(f"topk_preds: {topk_preds}")
        
        top_doc_ids = get_docids_from_a_list_of_preds(topk_preds)
        
        # A list of annoated evidence groups, each group has a set of (page_id, line_number) pairs.
        # Note, there could be duplicated evidence groups due to parallel annotation.
        evidence_sets = (
                    [
                        {
                            (evidence[2], str(evidence[3]))
                            for evidence in evidence_set
                        }
                        for evidence_set in items[i]['evidence']
                    ]
                    if items[i]['evidence']
                    else [set()]
                )
        
        # A list of document-level annotated evidence, i.e., each evidence group has a set of 'page_id'.
        evidence_docs = (
                        [
                            {evidence[2] for evidence in evidence_set}
                            for evidence_set in items[i]['evidence']
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
                                                                                                        evidence_sets=evidence_sets,
                                                                                                        evidence_docs=evidence_docs,                      
                                                                                                    )
        
    logger.info("Done ranking sentences in the top docs for each claim.")

    logger.info(
        "All examples sentence-level evidence hit ratio(fever-recall) @5: {}/{} = {:.12f}".format(
            hits, all_count, hits / (all_count + 1e-6)
        )
    )
    logger.info(
        "All examples document-level evidence hit ratio(fever-recall) @5: {}/{} = {:.12f}".format(
            hits, possible_count, hits / (possible_count + 1e-6)
        )
    )
    logger.info(
        "One-hop sentence-level evidence hit ratio(fever-recall) @5: {}/{} = {:.12f}".format(
            hits_1, all_count_1, hits_1 / (all_count_1 + 1e-6)
        )
    )
    logger.info(
        "One-hop document-level evidence Can hit ratio(fever-recall) @5: {}/{} = {:.12f}".format(
            hits_1, possible_count_1, hits_1 / (possible_count_1 + 1e-6)
        )
    )

    return hits / (all_count + 1e-6), hits / (possible_count + 1e-6), hits_1 / (all_count_1 + 1e-6), hits_1 / (possible_count_1 + 1e-6)

def main(args):
    bevers_srr_result = load_pickle(args.msrr_result_path)
    if args.debug:
        bevers_srr_result = bevers_srr_result[:10]
    
    if args.tune_params:
        best_merged_reranked_results = None
        best_hits = 0.0
        k=5
        
        if args.naive_merge:
            logger.info("Doing naive merging...")
            discount_iterable = list(i / 20 for i in range(21))
            for discount_factor in discount_iterable:
                logger.info("###################################################")
                logger.info("###################################################")
                logger.info(f"discount_factor: {discount_factor}")
                logger.info("###################################################")
                logger.info("###################################################")
                merged_reranked_results = merge_reranked_evi(items=bevers_srr_result,
                                                            naive_merge_discount_factor=discount_factor,
                                                            k=k,
                                                            )
                if 'label' in merged_reranked_results[0]:
                    s_all_hits, _, _, _ = retrieval_evaluator(items=merged_reranked_results)

                    if s_all_hits > best_hits:
                        best_hits = s_all_hits
                        best_discount_factor = discount_factor

                        best_merged_reranked_results = copy.deepcopy(merged_reranked_results)

                        logger.info("\n===================================================================")
                        logger.info("===================================================================")
                        logger.info("===================================================================")
                        logger.info(f"best_hits:              {best_hits}")
                        logger.info(f"best_discount_factor:   {best_discount_factor}")
            
            logger.info("\n*****************************************************************")
            logger.info("*****************************************************************")
            logger.info("*****************************************************************")
            logger.info(f"best_hits:              {best_hits}")
            logger.info(f"best_discount_factor:   {best_discount_factor}")
        else:
            logger.info("Doing conplex joint reranking...")
            msrr_merge_metric_list = ['product']
            mhth_list = [i * 0.001 for i in range(1, 10)]
            alpha = [0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            normalization = [True]
            weight_on_dense = [False]

            for msrr_merge_metric in msrr_merge_metric_list:
                for mhth in mhth_list:
                    if msrr_merge_metric == 'product' and mhth > 0.1:
                        continue
                    elif msrr_merge_metric != 'product' and mhth < 0.1:
                        continue
                    for norm in normalization:
                        for w_on_d in weight_on_dense:
                            for a in alpha:
                                logger.info("###################################################")
                                logger.info("###################################################")
                                logger.info(f"msrr_merge_metric: {msrr_merge_metric} ###########")
                                logger.info(f"mhth:              {mhth} ########################")
                                logger.info(f"norm:              {norm} ########################")
                                logger.info(f"w_on_d:            {w_on_d} ######################")
                                logger.info(f"a:                 {a} ###########################")
                                logger.info("###################################################")
                                logger.info("###################################################")

                                merged_reranked_results = merge_reranked_evi(
                                                                                items=bevers_srr_result,
                                                                                metric=msrr_merge_metric,
                                                                                mhth=mhth,
                                                                                alpha=a, 
                                                                                normalization=norm, 
                                                                                weight_on_dense=w_on_d,
                                                                                singleHopNumbers=args.singleHopNumbers,
                                                                                k=k,
                                                                            )

                                if 'label' in merged_reranked_results[0]:
                                    s_all_hits, _, _, _ = retrieval_evaluator(items=merged_reranked_results)

                                    if s_all_hits > best_hits:
                                        best_hits = s_all_hits
                                        best_msrr_merge_metric = msrr_merge_metric
                                        best_best_mhth = mhth
                                        best_normalization = norm
                                        best_weight_on_dense = w_on_d
                                        best_alpha = a

                                        best_merged_reranked_results = copy.deepcopy(merged_reranked_results)

                                        logger.info("\n===================================================================")
                                        logger.info("===================================================================")
                                        logger.info("===================================================================")
                                        logger.info(f"best_hits:              {best_hits}")
                                        logger.info(f"best_msrr_merge_metric: {best_msrr_merge_metric}")
                                        logger.info(f"best_best_mhth:         {best_best_mhth}")
                                        logger.info(f"best_normalization:     {best_normalization}")
                                        logger.info(f"best_weight_on_dense:   {best_weight_on_dense}")
                                        logger.info(f"best_alpha:             {best_alpha}")
        
            logger.info("\n*****************************************************************")
            logger.info("*****************************************************************")
            logger.info("*****************************************************************")
            logger.info(f"best_hits:              {best_hits}")
            logger.info(f"best_msrr_merge_metric: {best_msrr_merge_metric}")
            logger.info(f"best_best_mhth:         {best_best_mhth}")
            logger.info(f"best_normalization:     {best_normalization}")
            logger.info(f"best_weight_on_dense:   {best_weight_on_dense}")
            logger.info(f"best_alpha:             {best_alpha}")

    else:
        k=5
        logger.info("\n*****************************************************************")
        logger.info("*****************************************************************")
        logger.info("*****************************************************************")
        logger.info(f"msrr_merge_metric: {args.msrr_merge_metric}")
        logger.info(f"mhth:         {args.mhth}")
        logger.info(f"normalization:     {args.normalization}")
        logger.info(f"weight_on_dense:   {args.weight_on_dense}")
        logger.info(f"alpha:             {args.alpha}")
        if args.naive_merge:
            best_merged_reranked_results = merge_reranked_evi(
                                                                items=bevers_srr_result,
                                                                naive_merge_discount_factor=args.naive_merge_discount_factor,
                                                                k=k,
                                                            )
        else:
            best_merged_reranked_results = merge_reranked_evi(
                                                                items=bevers_srr_result,
                                                                metric=args.msrr_merge_metric,
                                                                mhth=args.mhth,
                                                                alpha=args.alpha,
                                                                normalization=args.normalization,
                                                                weight_on_dense=args.weight_on_dense,
                                                                singleHopNumbers=args.singleHopNumbers,
                                                                k=k,
                                                            )
                
    
    data_name = get_file_name(args.msrr_result_path)
    if 'label' in best_merged_reranked_results[0]:
        s_all_hits, _, _, _ = retrieval_evaluator(items=best_merged_reranked_results)
        merged_reranked_results_path = os.path.join(args.merged_reranked_results_dir, data_name + f'_msrr_{s_all_hits:.4f}.pkl')
    merged_reranked_results_path = os.path.join(args.merged_reranked_results_dir, data_name + f'_msrr.pkl')
    dump_pickle(best_merged_reranked_results, merged_reranked_results_path)
    
if __name__ == '__main__':
    make_directory(args.merged_reranked_results_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.merged_reranked_results_dir, f"merg_reranked_and_eval.log"))
    logger.info(args)
    main(args)
