import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import read_jsonl, save_jsonl, load_pickle, dump_pickle, process_evid, get_sentence_by_id, get_file_dir, make_directory
from src.IR.bm25 import BM25_Retriever

from typing import List, Any
from tqdm import tqdm
import copy

import argparse
from src.utils.args import prepare_logger
import logging

logger = logging.getLogger()

def define_args(parser):
    parser.add_argument('--data_path',
                        type=str,
                        required=False,
                        default="M3/data/FEVER_1/shares/train/all/0.jsonl"
                        )
            
    parser.add_argument('--index_dir',
                        type=str,
                        required=False,
                        default='M3/data/pyserini/index/sparse_term_frequency_embedding_noNER')

    parser.add_argument('--wikiSentence_path',
                        type=str,
                        required=False,
                        default='M3/data/pyserini/fever_wiki_sentence_dict_24950663.pkl')

    parser.add_argument('--cache_path',
                        type=str,
                        required=False,
                        default='M3/data/results/data_construction/dpr/temp/dpr_train_bm25_temp.pkl')
    
    parser.add_argument('--is_training_data',
                        type=str,
                        required=False,
                        default='True')
    
    parser.add_argument('--num_hard_negatives',
                        type=int,
                        required=True,
                        default=50)
    
    parser.add_argument('--multihop_mode',
                        type=str,
                        required=True,
                        default='two_hop',
                        help="Three difference modes of constructing multihop dense retrieval dataset: ['naive, iterative, two_hop'].")

class FeverToDPR_DataFormatter:
    def __init__(self,
                 data_path: str = None,
                 index_dir: str = None,
                 wikiSentence_path: str = None,
                 num_hard_negatives: int = 10,
                 is_training_data: bool = True,
                 add_multi_hop: bool = True,
                 add_single_hop: bool = True):
        
        logger.info(f"Loading dataset from: {data_path}")
        self.data_items = read_jsonl(data_path)
        logger.info("Done.")

        logger.info(f"Loading wikiSentence from: {wikiSentence_path}")
        self.wikiSentence = load_pickle(wikiSentence_path)
        logger.info("Done.")
        
        logger.info(f"Initializing hard-negative retriever with index from: {index_dir}")
        self.retriever = BM25_Retriever(index_dir=index_dir)
        logger.info("Done.")

        self.num_hard_negatives = num_hard_negatives
        logger.info(f"num_hard_negatives: {num_hard_negatives}")
        self.is_training_data = is_training_data
        logger.info(f"is_training_data: {self.is_training_data}.")

        self.add_multi_hop = add_multi_hop
        logger.info(f"add_multi_hop: {add_multi_hop}")

        self.add_single_hop = add_single_hop
        logger.info(f"add_single_hop: {add_single_hop}")

    def __get_positive_ctx_from_evidence_list(self, 
                                              item: dict, 
                                              evidence_list: List[Any]) -> dict:
        viewed_evidence = set()
        for evidence in evidence_list:
            evidence = (None, None, evidence[2], evidence[3])
            if evidence in viewed_evidence:
                continue
            viewed_evidence.add(evidence)
            title = process_evid(evidence[2])
            sentence_id = f"{evidence[2]}|#SEP#|{evidence[3]}"
            text = process_evid(get_sentence_by_id(sentence_id, self.wikiSentence))
            if text != '':
                item["positive_ctxs"].append({"title": title, 
                                            "text": text})
        return item


    def __get_hard_negative_ctx_with_bm25(self,
                                          item: dict):
        query = item['question']
        positives = [(pos['title'], pos['text']) for pos in item['positive_ctxs']]
        results = self.retriever.retrieve(query, self.num_hard_negatives + len(positives))[:self.num_hard_negatives * 2]
        
        hard_negatives = [(process_evid(r['id'].split('|#SEP#|')[0]), process_evid(get_sentence_by_id(r['id'], self.wikiSentence))) for r in results]

        checked_hard_negatives = []
        for hn in hard_negatives:
            if hn not in positives and (hn[1].strip().lower() not in query.strip().lower()):
                checked_hard_negatives.append(hn)
        
        item["hard_negative_ctxs"] = [{'title': hn[0], 'text': hn[1]} for hn in checked_hard_negatives][:self.num_hard_negatives]
        return item

    def __get_ctx_for_not_enough_info(self, item: dict):
        query = item['question']
        results = self.retriever.retrieve(query, self.num_hard_negatives + 1)
        
        hard_negatives = [(process_evid(r['id'].split('|#SEP#|')[0]), process_evid(get_sentence_by_id(r['id'], self.wikiSentence))) for r in results]

        sudo_pos_ctx = hard_negatives[0]
        item["positive_ctxs"] = [{'title': sudo_pos_ctx[0], 'text': sudo_pos_ctx[1]}]
        item["hard_negative_ctxs"] = [{'title': hn[0], 'text': hn[1]} for hn in hard_negatives[1:self.num_hard_negatives + 1]]
        return item

    def __drop_duplicated_evidence_group(self, evidence_group_list: List[List[Any]]) -> List[List[Any]]:
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

    def __fill_and_append_multihop_out_data_temp(self,
                                                query_evidence: List[str]=None,
                                                postive_evidence: List[str]=None,
                                                multihop_out_data_temp: dict=None,
                                                multihop_out_data_list: List[dict]=None) -> dict:
        title = process_evid(query_evidence[2])
        sentence_id = f"{query_evidence[2]}|#SEP#|{query_evidence[3]}"

        text = process_evid(get_sentence_by_id(sentence_id, self.wikiSentence))
        
        multihop_out_data_temp["question"] = ' -- '.join([multihop_out_data_temp["question"], f"{title} . {text}"]) if text != '' else multihop_out_data_temp["question"]
        # Construct a new question by concatnating the origianl query with first i evidence, and use the i+1th evidence as the positive context.
        # This is for better multi-hop inference.
        multihop_out_data = self.__get_positive_ctx_from_evidence_list(multihop_out_data_temp, [postive_evidence])

        if multihop_out_data['positive_ctxs'] != []:
            if len(multihop_out_data['positive_ctxs']) > 1:
                logger.info( f"Warning: len(multihop_out_data['positive_ctxs']) > 1, for multiHop_question: {multihop_out_data_temp['question']}")
            if self.is_training_data:
                multihop_out_data = self.__get_hard_negative_ctx_with_bm25(multihop_out_data)
        
            multihop_out_data_list.append(multihop_out_data)
        else:
            logger.info(f"Warning: multihop_out_data['positive_ctxs'] == [], for multiHop_question: {multihop_out_data_temp['question']}")

        return multihop_out_data_list
        
    
    def __get_multihop_data_list(self,
                                 item: dict, 
                                 multiHop_evidence_group: List[List[Any]]) -> List[dict]:
        
        multihop_out_data_list = []
        for evidence_list in multiHop_evidence_group:
            for i in range(len(evidence_list) - 1):
                multihop_out_data_temp = {"question": item['question'],
                                "answers": item['answers'],
                                "positive_ctxs": [],
                                "negative_ctxs": [],
                                "hard_negative_ctxs": [],
                                }
                
                multihop_out_data_list = self.__fill_and_append_multihop_out_data_temp(query_evidence=evidence_list[i],
                                                                                        postive_evidence=evidence_list[i+1],
                                                                                        multihop_out_data_temp=copy.deepcopy(multihop_out_data_temp),
                                                                                        multihop_out_data_list=copy.deepcopy(multihop_out_data_temp))

        return multihop_out_data_list
    
    def __get_twohop_data_list(self,
                                 item: dict, 
                                 multiHop_evidence_group: List[List[Any]]) -> List[dict]:
        
        multihop_out_data_list = []
        for evidence_list in multiHop_evidence_group:
            if len(evidence_list) > 2:
                logger.info(f"WARNING: len(evidence_list) > 2")
                logger.info(f"item: {item}")
                continue

            multihop_out_data_temp = {"question": item['question'],
                                        "answers": item['answers'],
                                        "positive_ctxs": [],
                                        "negative_ctxs": [],
                                        "hard_negative_ctxs": [],
                                        }

            multihop_out_data_list = self.__fill_and_append_multihop_out_data_temp(query_evidence=evidence_list[0],
                                                                                        postive_evidence=evidence_list[1],
                                                                                        multihop_out_data_temp=copy.deepcopy(multihop_out_data_temp),
                                                                                        multihop_out_data_list=multihop_out_data_list)
            multihop_out_data_list = self.__fill_and_append_multihop_out_data_temp(query_evidence=evidence_list[1],
                                                                                        postive_evidence=evidence_list[0],
                                                                                        multihop_out_data_temp=copy.deepcopy(multihop_out_data_temp),
                                                                                        multihop_out_data_list=multihop_out_data_list)
                
        return multihop_out_data_list

    def __get_naive_multihop_data_list(self, item: dict, 
                                        multiHop_evidence_group: List[List[Any]]) -> List[dict]:
        multihop_out_data_list = []
        multihop_out_data = {"question": item['question'],
                                    "answers": item['answers'],
                                    "positive_ctxs": [],
                                    "negative_ctxs": [],
                                    "hard_negative_ctxs": []}
        evidence_list_all = []
        for evidence_list in multiHop_evidence_group:
            for evidence in evidence_list:          
                evidence_list_all.append(evidence)
        multihop_out_data = self.__get_positive_ctx_from_evidence_list(multihop_out_data, evidence_list_all)

        if multihop_out_data['positive_ctxs'] != []:
            if self.is_training_data:
                multihop_out_data = self.__get_hard_negative_ctx_with_bm25(multihop_out_data)

            multihop_out_data_list.append(multihop_out_data)
        else:
            logger.info(f"Warning: multihop_out_data['positive_ctxs'] == [], for multiHop_question: {item['question']}")

        return multihop_out_data_list
        
    def transform_data_format(self, 
                              worker_id: int = -1,
                              cache_path: str = '',
                              multihop_mode: str=None,
                              ADD_NOT_ENOUTH_INFO: bool=False) -> List[dict]:
        
        
        worker_id = 'Solo' if worker_id == -1 else worker_id
        
        cache_dir = make_directory(get_file_dir(cache_path))
        progress_path = cache_path.replace('.pkl', '_progress.pkl')
        total_task = len(self.data_items)
        
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")
            data_list = load_pickle(cache_path)
            cached_task = load_pickle(progress_path)
            logger.info('Done.')
            logger.info(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
        else:
            data_list = []
            cached_task = -1
            
        
        logger.info(f">>Worker {worker_id}: Resuming from: {cached_task}/{total_task}.")
        
        for i in tqdm(range(total_task), total=total_task, 
                    desc="Transforming FEVER data format to DPR ..."):

            item = self.data_items[i]
            question = item['claim']
            answer = item['label']
            out_data = {"question": question,
                        "answers": [answer],
                        "positive_ctxs": [],
                        "negative_ctxs": [],
                        "hard_negative_ctxs": []}

            if i <= cached_task:
                continue

            if self.data_items[i]['verifiable'].upper() != 'NOT VERIFIABLE': 
                if not self.add_single_hop and not self.add_multi_hop:
                    logger.info(f"Countinued! Position: 1")
                    continue
                    
                single_evidence_group = []
                multiHop_evidence_group = []
                logger.info(f">>Worker {worker_id}: Dropping duplicated evidence group...")
                evidence_group_list = self.__drop_duplicated_evidence_group(item['evidence'])
                logger.info(f">>Worker {worker_id}: Done.")
                for evidence_group in evidence_group_list:
                    if len(evidence_group) < 1:
                        continue
                    
                    if self.add_single_hop and len(evidence_group) > 1:
                        single_evidence_group.append(evidence_group[0])

                    if self.add_multi_hop and len(evidence_group) > 1:
                        if multihop_mode == "two_hop":
                            if len(evidence_group) == 2:
                                multiHop_evidence_group.append(evidence_group)
                        else:
                            multiHop_evidence_group.append(evidence_group)     
                
                if self.add_single_hop:
                    logger.info(f"len(single_evidence_group): {len(single_evidence_group)}")
                    logger.info(f"single_evidence_group: {single_evidence_group}")
                    singlehop_out_data = self.__get_positive_ctx_from_evidence_list(out_data, single_evidence_group)

                    if singlehop_out_data['positive_ctxs'] != []:
                        if self.is_training_data:
                            singlehop_out_data = self.__get_hard_negative_ctx_with_bm25(singlehop_out_data)
                        data_list.append(singlehop_out_data)
                    else:
                        logger.info(f"Warning: singlehop_out_data['positive_ctxs'] == [], for singlehop_out_data: {singlehop_out_data}")
                
                if self.add_multi_hop:
                    logger.info(f"len(multiHop_evidence_group): len(multiHop_evidence_group)")
                    logger.info(f"multiHop_evidence_group: {multiHop_evidence_group}")
                    if multihop_mode == "naive":
                        logger.info(f">>Worker {worker_id}: Getting naitve multihop data list...")
                        multihop_out_data_list = self.__get_naive_multihop_data_list(out_data, multiHop_evidence_group)
                        logger.info(f">>Worker {worker_id}: Done.")
                    elif multihop_mode == "iterative":
                        logger.info(f">>Worker {worker_id}: Getting iterative multihop data list...")
                        multihop_out_data_list = self.__get_multihop_data_list(out_data, multiHop_evidence_group)
                        logger.info(f">>Worker {worker_id}: Done.")
                    elif multihop_mode == "two_hop":
                        logger.info(f">>Worker {worker_id}: Getting two-hop data list...")
                        multihop_out_data_list = self.__get_twohop_data_list(out_data, multiHop_evidence_group)
                        logger.info(f">>Worker {worker_id}: Done.")
                    else:
                        raise Exception(f"Unsupported multihop_mode: {multihop_mode}. Support multihop_mode: ['naive', 'iterative', 'two_hop']")
                    data_list.extend(multihop_out_data_list)
            elif ADD_NOT_ENOUTH_INFO:
                not_enough_info_out_data = self.__get_ctx_for_not_enough_info(out_data)

                if not_enough_info_out_data['positive_ctxs'] != []:
                    data_list.append(not_enough_info_out_data)
                else:
                    logger.info(f"Warning: not_enough_info_out_data['positive_ctxs'] == [], for not_enough_info_out_data: {not_enough_info_out_data}")
            else:
                logger.info(f"Countinued! Position: 2")
                continue
            
            if i % 100 == 0:
                logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                dump_pickle(data_list, cache_path)
                dump_pickle(i, progress_path)
                logger.info(f">>Worker {worker_id}: Done.")
                
        logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
        dump_pickle(data_list, cache_path)
        dump_pickle(i, progress_path)
        logger.info(f">>Worker {worker_id}: All Done.")

        if 'temp' in cache_path.lower():
            logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path.replace('.pkl', '.jsonl')}")
            save_jsonl(data_list, cache_path.replace('.pkl', '.jsonl'))
            logger.info(f">>Worker {worker_id}: All Done for sure.")

        return


def main(data_path: str = None,
         index_dir: str = None,
         wikiSentence_path: str = None,
         cache_path: str = None,
         num_hard_negatives: int = 50,
         is_training_data: bool = True,
         add_multi_hop: bool = True,
         add_single_hop: bool = True,
         multihop_mode: bool = None,
         ADD_NOT_ENOUTH_INFO: bool = False):

    data_formatter = FeverToDPR_DataFormatter(data_path=data_path, 
                                              index_dir=index_dir, 
                                              wikiSentence_path=wikiSentence_path, 
                                              num_hard_negatives=num_hard_negatives, 
                                              is_training_data=is_training_data,
                                              add_multi_hop=add_multi_hop,
                                              add_single_hop=add_single_hop)
    
    data_formatter.transform_data_format(cache_path=cache_path, 
                                         multihop_mode=multihop_mode, 
                                         ADD_NOT_ENOUTH_INFO=ADD_NOT_ENOUTH_INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build DPR datasets with FEVER dataset, WikiSentence corpus, and BM25 hard-negative sampler.')
    define_args(parser)
    args = parser.parse_args()
    prepare_logger(logger, debug=False, save_to_file=os.path.join(get_file_dir(args.cache_path), "fever_to_dpr.log"))


    data_path = args.data_path
    index_dir = args.index_dir
    wikiSentence_path = args.wikiSentence_path
    cache_path = args.cache_path

    num_hard_negatives = args.num_hard_negatives
    if args.is_training_data.lower() in {'true', 't', '1'}:
        is_training_data = True
    elif args.is_training_data.lower() in {'false', 'f', '0'}:
        is_training_data = False
    else:
        raise Exception(f"Error: Invalid args.is_training_data: {args.is_training_data}")

    add_multi_hop = False
    multihop_mode = args.multihop_mode
    add_single_hop = True
    ADD_NOT_ENOUTH_INFO = False
    
    logger.info(f"add_multi_hop: {add_multi_hop}")
    logger.info(f"multihop_mode: {multihop_mode}")
    logger.info(f"add_single_hop: {add_single_hop}")
    logger.info(f"ADD_NOT_ENOUTH_INFO: {ADD_NOT_ENOUTH_INFO}")

    main(data_path=data_path, index_dir=index_dir, wikiSentence_path=wikiSentence_path, cache_path=cache_path, 
         num_hard_negatives=num_hard_negatives, is_training_data=is_training_data, add_multi_hop=add_multi_hop, 
         add_single_hop=add_single_hop, multihop_mode=multihop_mode, ADD_NOT_ENOUTH_INFO=ADD_NOT_ENOUTH_INFO)
