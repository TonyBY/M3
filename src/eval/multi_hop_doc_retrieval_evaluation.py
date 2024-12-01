import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import get_file_dir, make_directory, load_pickle, dump_pickle, \
                                 process_evid, get_sentence_by_id

from src.utils.config import parser
from src.utils.args import prepare_logger

import logging
from typing import List, Tuple, Any
from tqdm import tqdm

from SetSimilaritySearch import SearchIndex

args = parser.parse_args()
logger = logging.getLogger()

def __get_docids_in_corpus(wiki_line_dict: dict, sep: str='|#SEP#|') -> List[str]:
    docid_corpus = set()
    for key in wiki_line_dict.keys():
        docid_corpus.add(key.split(sep)[0])
    return list(docid_corpus)

def __retrieve_docids_by_hyperlink(hyperlink: str=None, 
                                    docid_corpus_index=None,  
                                    topk: int=10) -> List[dict]:
    normalized_hyperlink = hyperlink.strip().lower().split()

    results = docid_corpus_index.query(normalized_hyperlink)

    sorted_results = sorted(
                        results, 
                        key=lambda x: x[1],
                        reverse=True
                    )[:topk]

    return sorted_results

def __get_docids_from_a_pred_sentids(pred_sentids: List[str]=None, 
                             docid_corpus: List[str]=None,
                             docid_corpus_index=None,
                             sep: str='|#SEP#|',
                             topk: int=3,
                             wiki_line_dict: dict=None) -> List[str]:
    
    multi_hop_docid_corpus = []
    
    hyperlinks = set()
    for pred_sentid in pred_sentids:
        doc_id = pred_sentid.split(sep)[0]
        hyperlinks.add(doc_id)
        hyperlink_all = get_sentence_by_id(pred_sentid, wiki_line_dict).split('\t')
        if hyperlink_all[0].isdigit():
            hyperlinks.update(hyperlink_all[2:])
        else:
            hyperlinks.update(hyperlink_all[1:])
        
    
    logger.debug(f"hyperlinks: {hyperlinks}")
    for hl in hyperlinks:
        logger.debug("================")
        logger.debug(f"hl: {hl}")
        topk_doc_idx = __retrieve_docids_by_hyperlink(hyperlink=hl, 
                                                        docid_corpus_index=docid_corpus_index, 
                                                        topk=topk,
                                                        )
        topk_docids = [docid_corpus[idx[0]] for idx in topk_doc_idx]
        logger.debug(f"topk_docids: {topk_docids}")
        multi_hop_docid_corpus.extend(topk_docids)
        
    return list(set(multi_hop_docid_corpus))

def __drop_duplicated_evidence_group(evidence_group_list: List[List[Any]]) -> List[List[Any]]:
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

def __get_hopnum_to_itemidx_map(fever_data: List[dict]) -> dict:
    hop_idx_dict = {} # Indecies of items that have a certain evidence hop. 
    hop_count_dict = {} # count of evidence group of different evidence hop.
    valid_hop_count_dict = {} # Count of Items of different minimum evidence hop.
    valid_hop_idx_dict = {} # # Indecies of items that have a certain evidence hop. 
    super_multihop_cnt=0
    impossible_items = []
    multi_hop_cnt = 0
    valid_multi_hop_cnt = 0
    for idx, item in tqdm(enumerate(fever_data)):
        evidence_group_list = __drop_duplicated_evidence_group(item['evidence'])
        if any(len(evidence_group) <= 1  for evidence_group in evidence_group_list) or item['verifiable'] == 'NOT VERIFIABLE':
            continue
        
        min_hop = min([len(evidence_group) for evidence_group in evidence_group_list])
        max_hop = max([len(evidence_group) for evidence_group in evidence_group_list])
    
        if max_hop > 1 and item['verifiable'] != 'NOT VERIFIABLE':
            multi_hop_cnt+=1
            if min_hop > 1:
                valid_multi_hop_cnt += 1
                
        if min_hop > 2:
            super_multihop_cnt+=1
        if min_hop > 5:
            impossible_items.append(item)
            
        if min_hop not in valid_hop_count_dict:
            valid_hop_count_dict[min_hop] = 1
            valid_hop_idx_dict[min_hop] = [idx]
        else:
            valid_hop_count_dict[min_hop] += 1
            valid_hop_idx_dict[min_hop].append(idx)
            
        for evidence_group in evidence_group_list:
            if len(evidence_group) in hop_count_dict:
                hop_count_dict[len(evidence_group)] += 1
                hop_idx_dict[len(evidence_group)].append(idx)
            else:
                hop_count_dict[len(evidence_group)] = 1
                hop_idx_dict[len(evidence_group)] = [idx]
    
    logger.info(f"multi_hop_cnt: {multi_hop_cnt}")
    logger.info(f"valid_multi_hop_cnt: {valid_multi_hop_cnt}")
    logger.info(f"super_multihop_cnt: {super_multihop_cnt}")
    logger.info(f"len(impossible_items): {len(impossible_items)}")
    logger.info(f"valid_hop_count_dict: {valid_hop_count_dict}")
                
    return valid_hop_idx_dict

def __get_multihop_examples_by_maxhop(valid_hop_idx_dict: dict=None, 
                                      maxhop: int=2, 
                                      fever_data: List[dict]=None) -> List[dict]:
    valid_multi_hop_items = []
    viewed_idx = set()
    for hop in tqdm(range(maxhop + 1)):        
        if hop not in valid_hop_idx_dict.keys():
            continue
            
        for idx in tqdm(valid_hop_idx_dict[hop]):
            if idx in viewed_idx:
                continue
                
            viewed_idx.add(idx)
            valid_multi_hop_items.append(fever_data[idx])
            
    return valid_multi_hop_items

def __do_multihop_doc_retrieval(valid_multi_hop_items: List[dict]=None, 
                                sep: str='|#SEP#|',
                                topk: int=5,
                                docid_corpus: List[str]=None,
                                docid_corpus_index=None,
                                wiki_line_dict: dict=None) -> List[dict]:
    
    multi_hop_doc_retrieval_results = []
    for item in tqdm(valid_multi_hop_items, desc="Doing multi-hop doc retrieval..."):
        pred_sentids = [sent[0] + sep + sent[1] for sent in item['reranked_context']]
        logger.debug(f"pred_sentids: {pred_sentids}")

        item['multi_hop_docids'] = __get_docids_from_a_pred_sentids(pred_sentids=pred_sentids, 
                                                                 docid_corpus=docid_corpus,
                                                                 docid_corpus_index=docid_corpus_index,
                                                                 sep=sep,
                                                                 topk=topk,
                                                                 wiki_line_dict=wiki_line_dict)
        multi_hop_doc_retrieval_results.append(item)
        
    return multi_hop_doc_retrieval_results

def __get_docid_from_evi(evi):
    return evi[2]

def __check_doc_level_hit(evidence_groups: List[List[Tuple]]=None, 
                          topk_doc_ids:List[str]=None) -> bool:

    page_id_groups = [
        [__get_docid_from_evi(evidence) for evidence in evidence_group]
        for evidence_group in evidence_groups
    ]

    all_hit = False
    for page_id_group in page_id_groups:    
        all_hit = all_hit or all(
            [page_id in topk_doc_ids for page_id in page_id_group]
        )
        if all_hit:
            break
        
    return all_hit

def __eval_doc_recall(retrievl_results: List[dict]=None, 
                     ) -> float:
    
    hits = 0
    total = 0
    
    for i, item in tqdm(enumerate(retrievl_results), 
                    total=len(retrievl_results),
                    desc=f'Evaluating doc-level retrieval recall...'):
    
        if item['label'] == 'NOT ENOUGH INFO':
#             hits += 1
            continue
            
        total += 1
            
        pred_doc_ids = item['multi_hop_docids']

        logger.info("====================")
        logger.info(f"item['evidence']: {item['evidence']}")
        logger.info("--------------------")
        logger.info(f"pred_doc_ids: {pred_doc_ids}")
            
        if __check_doc_level_hit(evidence_groups=item['evidence'], 
                                  topk_doc_ids=pred_doc_ids):
            hits += 1

    logger.info(f"Number of docs: {total}")
    return hits/total

def main(args):
    sufficiency_checking_results = load_pickle(args.sufficiency_checking_results_path)
    if args.debug:
        sufficiency_checking_results = sufficiency_checking_results[:100]
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    wiki_line_dict_dir = get_file_dir(args.wiki_line_dict_pkl_path)

    docid_corpus_path = os.path.join(wiki_line_dict_dir, "docid_corpus.pkl")
    if os.path.exists(docid_corpus_path):
        logger.info(f"Loading docid_corpus from: {docid_corpus_path}")
        docid_corpus = load_pickle(docid_corpus_path)
    else:
        logger.info("Constructing docid_corpus...")
        docid_corpus = __get_docids_in_corpus(wiki_line_dict)
        logger.info("Done.")
        dump_pickle(docid_corpus, docid_corpus_path)
    len(docid_corpus), docid_corpus[:10]
    
    docid_corpus_index_path = os.path.join(wiki_line_dict_dir, f"docid_corpus_index_simfun-{args.similarity_func_name}_th-{args.similarity_threshold}.pkl")
    if os.path.exists(docid_corpus_index_path):
        logger.info(f"Loading index from: {docid_corpus_index_path}")
        docid_corpus_index = load_pickle(docid_corpus_index_path)
    else:
        splited_docid_corpus_path = os.path.join(wiki_line_dict_dir, f"splited_docid_corpus.pkl")
        if os.path.exists(splited_docid_corpus_path):
            logger.info(f"Loading splited_docid_corpus from: {splited_docid_corpus_path}")
            splited_docid_corpus = load_pickle(splited_docid_corpus_path)
        else:
            logger.info(f"Constructing splited_docid_corpus...")
            splited_docid_corpus = [process_evid(docid.strip().lower()).split() for docid in tqdm(docid_corpus)]
            logger.info("Done.")
            dump_pickle(splited_docid_corpus, splited_docid_corpus_path)

        logger.info(f"Start building index with simfun: {args.similarity_func_name}, threshod: {args.similarity_threshold}...")
        docid_corpus_index = SearchIndex(splited_docid_corpus, 
                                        similarity_func_name=args.similarity_func_name, 
                                        similarity_threshold=args.similarity_threshold)
        logger.info(f"Done.")
        logger.info(f"Dumping index to: {docid_corpus_index_path}")
        dump_pickle(docid_corpus_index, docid_corpus_index_path)

    valid_hop_idx_dict = __get_hopnum_to_itemidx_map(sufficiency_checking_results)

    valid_multi_hop_items = __get_multihop_examples_by_maxhop(valid_hop_idx_dict=valid_hop_idx_dict, 
                                                            maxhop=args.maxhop, 
                                                            fever_data=sufficiency_checking_results)
    logger.info(f"len(valid_multi_hop_items): {len(valid_multi_hop_items)}")

    multi_hop_doc_retrieval_results = __do_multihop_doc_retrieval(valid_multi_hop_items=valid_multi_hop_items, 
                                                                sep='|#SEP#|',
                                                                topk=args.mdr_topk,
                                                                docid_corpus=docid_corpus,
                                                                docid_corpus_index=docid_corpus_index,
                                                                wiki_line_dict=wiki_line_dict)
    
    multi_hop_doc_retrieval_results_path = os.path.join(args.multihop_doc_retrieval_dir, 'multi_hop_doc_retrieval_results.pkl')
    logger.info(f"len(multi_hop_doc_retrieval_results): {len(multi_hop_doc_retrieval_results)}")
    dump_pickle(multi_hop_doc_retrieval_results, multi_hop_doc_retrieval_results_path)
    
    doc_recall = __eval_doc_recall(retrievl_results=multi_hop_doc_retrieval_results)
    logger.info(f"doc_recall: {doc_recall}")


if __name__ == "__main__":
    make_directory(args.multihop_doc_retrieval_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.multihop_doc_retrieval_dir, "multihop_doc_retrieval.log"))
    logger.info(args)
    main(args)
    