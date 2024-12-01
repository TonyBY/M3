import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from typing import List, Optional
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from src.utils.data_utils import read_jsonl, save_jsonl, merge_mp_results, split_list, \
                                 get_file_dir, make_directory, load_pickle, get_sentence_by_id, \
                                 get_file_name, process_evid
from src.IR.base import Retriever
import multiprocessing as mp
import pickle
import math
import logging

from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()
sep='|#SEP#|'                    


class BM25_Retriever(Retriever):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 index_dir: str = None,
                 output_path: Optional[str] = None):
        self.k1 = k1
        self.b = b
        self.index_dir = index_dir
        self.output_path = output_path

    def batch_retrieve(self, items: List[dict], topk=1000, worker_id=0) -> List[dict]:
        cache_path = self.output_path
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")

            data_list = []
            if os.path.exists(cache_path):
                data_list = load_pickle(cache_path) 

            logger.info('Done.')
            logger.info(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
        else:
            data_list = []
            
        cached_task = len(data_list)
        total_task = len(items)

        logger.info(f">>Worker {worker_id}: Resuming from: {cached_task}/{total_task}.")

        for i in tqdm(range(total_task), total=total_task, desc=f">>Worker {worker_id} is retrieving for every query..."):
            if i < cached_task:
                continue
            query = items[i]["claim"]
            result_list = self.retrieve(query, topk=topk)
            items[i]["context"] = result_list
            data_list.append(items[i])

            if i % 10000 == 0:
                logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                with open(cache_path, 'wb') as file:
                    pickle.dump(data_list, file)
                logger.info(f">>Worker {worker_id}: Done.")
                
        logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
        with open(cache_path, 'wb') as file:
            pickle.dump(data_list, file)
        logger.info(f">>Worker {worker_id}: Done.")
        return data_list
    
    def get_insuf_sent_ids(self, item: dict=None, 
                           srr_th: float=1.0, 
                           sf_th: float=1.0) -> List[str]:
        """
        params: 
            item: a data example, a dictionary with keys: ['id', 'verifiable', 'label', 'claim', 'evidence', 'context', 'reranked_context', 'sufficiency_checking_results']
            srr_th: sentnence reranking score threshold, used for multi_hop evidence selecting.
            sf_th: sufficiency checking score threshold, used for multi_hop evidence selecting.
        output:
            sent_ids: a list of sentence id strings that will need to do a secent round of retrieval for the secend hop evidence.
        """

        if 'sufficiency_checking_results' in item:
            logger.info("Concatenating evidence from 'sufficiency_checking_results'.")
            if any([int(evi[3]) == 2 and float(evi[4]) > sf_th and float(evi[2]) > srr_th for evi in item['sufficiency_checking_results']]) \
                or all([int(evi[3]) == 1 and float(evi[4]) > sf_th for evi in item['sufficiency_checking_results']]):
                return []
            else:
                return [sep.join([evi[0], evi[1]]) for evi in item['sufficiency_checking_results'] if int(evi[3]) == 0 or float(evi[4]) < sf_th]
        else:
            logger.info("Concatenating evidence from 'reranked_context'.")
            return [sep.join([evi[0], evi[1]]) for evi in item['reranked_context'] if float(evi[2]) < srr_th]
    

    def multi_hop_batch_retrieve(self, items: List[dict], 
                                 topk: int=1000, 
                                 worker_id=0,
                                 wiki_line_dict: dict=None) -> List[dict]:
        worker_id = 'Solo' if worker_id == -1 else worker_id

        cache_path = self.output_path
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")

            data_list = []
            if os.path.exists(cache_path):
                data_list = load_pickle(cache_path) 

            logger.info('Done.')
            logger.info(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
        else:
            data_list = []
            
        cached_task = len(data_list)
        total_task = len(items)

        logger.info(f">>Worker {worker_id}: Resuming from: {cached_task}/{total_task}.")

        for i in tqdm(range(total_task), total=total_task, desc=f">>Worker {worker_id} is retrieving for every query..."):
            if i < cached_task:
                continue

            insuf_sent_ids = self.get_insuf_sent_ids(items[i], 
                                                     srr_th=args.srr_th, 
                                                     sf_th=args.sf_th)
            
            items[i]['multihop_context'] = []
            if len(insuf_sent_ids) != 0:
                lines = []
                for sentid in insuf_sent_ids:
                    logger.debug(f"sentid: {sentid}")

                    evi_text = get_sentence_by_id(sentid, wiki_line_dict)
                    logger.debug(f"evi_text: {evi_text}")

                    if evi_text == '':
                        logger.info(f"WARNING: Sentence: {sentid} not in wiki_line_dict.")
                        continue

                    if '\t' in evi_text:
                        if evi_text.split('\t')[0].isdigit():
                            lines.append([process_evid(sentid.split(sep)[0]), process_evid(evi_text.split('\t')[1])])
                        else:
                            lines.append([process_evid(sentid.split(sep)[0]), process_evid(evi_text)])
                    else:
                        lines.append([process_evid(sentid.split(sep)[0]), process_evid(evi_text)])

                
                logger.debug(f"lines: {lines}")

                if lines == []:
                    logger.info(f"WARNING: No evidence is available.")
                    continue

                logger.debug(f"insuf_sent_ids: {insuf_sent_ids}")
                logger.debug(f"lines: {lines}")

                batch_q = [" -- ".join([items[i]["claim"], evi[1]]) for evi in lines]
                logger.debug(f"batch_q: {batch_q}")

                batch_context_list = []
                for query in batch_q:
                    result_list = self.retrieve(query, topk=min(topk * 5, total_task))
                    batch_context_list.append(result_list) 

                sorted_batch_context_list = sorted([ctx for ctx_list in batch_context_list for ctx in ctx_list], 
                                                    key=lambda ctx: float(ctx['score']), 
                                                    reverse=True)
                 
                viewed_sent_ids = set(insuf_sent_ids)
                for j in range(min(topk, len(sorted_batch_context_list))):
                    ctx = sorted_batch_context_list[j]
                    if ctx['id'] not in viewed_sent_ids:
                        viewed_sent_ids.add(ctx['id'])
                        items[i]['multihop_context'].append(ctx)

                logger.info(f"len(items[i]['multihop_context']): {len(items[i]['multihop_context'])}")
                
            data_list.append(items[i])

            if i % 10000 == 0:
                logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                with open(cache_path, 'wb') as file:
                    pickle.dump(data_list, file)
                logger.info(f">>Worker {worker_id}: Done.")
                
        logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
        with open(cache_path, 'wb') as file:
            pickle.dump(data_list, file)
        logger.info(f">>Worker {worker_id}: Done.")
        return data_list
    

    def retrieve(self, query: str, topk: int = 1000) -> List[dict]:
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(k1=self.k1, b=self.b)
        hits = searcher.search(query, k=topk,
                               remove_dups=True)
        data_list = []
        for j in range(len(hits)):
            data = {}
            data["id"] = str(hits[j].docid)
            data["score"] = str(hits[j].score)
            data_list.append(data)
        return data_list

def main(data_list, index_dir, output_path, max_num_process=20, k1=1.6, b=0.75, topk=2048, wiki_line_dict: dict=None):
    file_size = len(data_list)
    num_workers = min(mp.cpu_count(), max_num_process)
    num_workers = num_workers if file_size // num_workers > 1 else 1

    if num_workers > 1:
        cache_dir = os.path.join(get_file_dir(output_path) + 'cache')
        make_directory(cache_dir)
        logger.info(f"{cache_dir}: {cache_dir}")

        chunk_size = math.ceil(file_size / num_workers)
        chunks = split_list(data_list, chunk_num=num_workers)

        logger.info(f"num_workers: {num_workers}")
        logger.info(f"file_size: {file_size}")
        logger.info(f"chunk_size: {chunk_size}")
        logger.info(f"number of chunks: {len(chunks)}")

        results = []
        pool = mp.Pool(num_workers)
        for i in range(len(chunks)):
            bm25_retriever = BM25_Retriever(k1=k1, b=b, index_dir=index_dir, output_path=f"{cache_dir}/{i}.pkl")
            if wiki_line_dict:
                proc = pool.apply_async(bm25_retriever.multi_hop_batch_retrieve, (chunks[i],), dict(topk=topk, worker_id=i, wiki_line_dict=wiki_line_dict))
            else:
                proc = pool.apply_async(bm25_retriever.batch_retrieve, (chunks[i],), dict(topk=topk, worker_id=i))
            results.append(proc)
        pool.close()
        pool.join()

        retrieved_items = merge_mp_results(results)
    else:
        # Serial method:  
        bm25_retriever = BM25_Retriever(k1=k1, b=b, index_dir=index_dir, output_path=output_path.replace('.jsonl', '.pkl'))
        if wiki_line_dict:
            retrieved_items = bm25_retriever.multi_hop_batch_retrieve(data_list, topk=topk, worker_id='solo', wiki_line_dict=wiki_line_dict)
        else:
            retrieved_items = bm25_retriever.batch_retrieve(data_list, topk=topk, worker_id='solo')
    logger.info(f"len(retrieved_items): {len(retrieved_items)}")
    return retrieved_items

if __name__ == "__main__":
    k1 = 1.6
    b = 0.75
    topk = 2048
    max_num_process = 1

    data_name = get_file_name(args.data_path)

    if args.multi_hop_sparse_retrieval:
        log_file = 'multi_hop_bm25_searcher'
        output_file_name = f'{data_name}_bm25_multihop_search_result_top{topk}.pkl'
        logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
        wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)
    else:
        log_file = 'single_hop_bm25_searcher'
        output_file_name = f'{data_name}_bm25_singlehop_search_result_top{topk}.pkl'
        wiki_line_dict = None

    if args.debug:
        output_file_name = "DEBUG_" + output_file_name

    if args.multi_hop_sparse_retrieval_dir:
        output_dir = args.multi_hop_sparse_retrieval_dir
    elif args.output_path:
        output_dir = get_file_dir(args.output_path)
    else:
        raise Exception("Error, args.multi_hop_sparse_retrieval_dir and args.output_path cannot both be empty string.")
    
    make_directory(output_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(output_dir, f"{log_file}.log"))
    logger.info(args)

    logger.info(f"k1: {k1}")
    logger.info(f"b: {b}")
    logger.info(f"topk: {topk}")
    logger.info(f"max_num_process: {max_num_process}")

    output_path = args.output_path if args.output_path else os.path.join(output_dir, output_file_name)
    logger.info(f"output_path: {output_path}")

    logger.info(f"Loading queries from: {args.data_path}")
    if args.data_path.endswith('.jsonl'):
        data_list = read_jsonl(args.data_path)
    elif args.data_path.endswith('.pkl'):
        data_list = load_pickle(args.data_path)
    logger.info(f"len(data_list): {len(data_list)}")

    if args.debug:
        data_list = data_list[:5]

    main(data_list, args.index_dir, output_path, max_num_process=max_num_process, k1=k1, b=b, topk=topk, wiki_line_dict=wiki_line_dict)
