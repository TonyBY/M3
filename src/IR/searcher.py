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

from typing import List
from tqdm import tqdm
from dataclasses import dataclass
import faiss
import math
import numpy as np
import multiprocessing as mp
import logging
import time

import torch
import transformers

from src.models.joint_retrievers import BiEncoderRetriever, SingleEncoderRetriever
from src.utils.model_utils import load_saved_model
from src.utils.data_utils import timer, get_file_dir, get_file_name, make_directory, read_jsonl, save_jsonl, load_pickle, \
                                    dump_pickle, split_list, merge_mp_results, get_sentence_by_id, process_evid
from src.utils.config import parser
from src.utils.args import prepare_logger
from src.index.indexer import Indexer

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

sep = '|#SEP#|'


@dataclass
class DenseSearchResult:
    docid: str
    score: float


class Searcher(Indexer):
    def __init__(self, 
                 model,
                 tokenizer_name_or_path:str=None, 
                 max_length:int=512,
                 sentence_type:str="query",
                 sentence_corpus_path:str=None,
                 index=None,
                 index_dir:str=None,
                 index_type:str=None,
                 query_encoder_device=None,
                 index_in_gpu: bool = False,
                 docids: List[str] = None,
                 ):
                
        logger.info(f"query_encoder_device: {query_encoder_device}")
        logger.info(f"sentence_type: {sentence_type}")

        super().__init__(model,
                         tokenizer_name_or_path=tokenizer_name_or_path, 
                         max_length=max_length,
                         sentence_type=sentence_type,
                         sentence_corpus_path=sentence_corpus_path,
                         index_dir=index_dir,
                         device=query_encoder_device,
                         )

        if docids:
            self.docids = docids
        else:
            self.docids = self.load_docids()
        
        if index:
            self.index = index
        else:
            index_path = os.path.join(self.index_dir, f'{index_type}_index')
            logger.info(f"Loading index from: {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info("Done.")
        
        logger.info(f"index_in_gpu: {index_in_gpu}")
        logger.info(f"self.encoder.device.type: {self.encoder.device.type}")
        logger.info(f"type(self.encoder): {type(self.encoder)}")
        if index_in_gpu and self.encoder.device.type == 'cuda':
            if 'HNSW' in str(type(self.index)).upper():
                logger.info(f"Faiss HNSW index does not support running on gpu.")
                logger.info(f"Using cpu for searching.")
            else:
                # Note, faiss HNSW index does not support running on gpu. 
                # It will just copy the index to another CPU index (the default behavior for indexes it does not recognize).
                res = faiss.StandardGpuResources()
                logger.info("Loading index to gpu.")
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index) # Use the gpu of index 0.
                logger.info("Done.")
        else:
            logger.info(f"Using cpu for searching.")
    
    @timer
    def load_docids(self) -> List[str]:
        logger.info(f"Loading docids from: {self.doc_id_path}")
        id_f = open(self.doc_id_path, 'r')
        docids = [line.rstrip() for line in id_f.readlines()]
        id_f.close()
        logger.info(f"Done.")
        return docids

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
        elif 'reranked_context' in item:
            logger.info("Concatenating evidence from 'reranked_context'.")
            return [sep.join([evi[0], evi[1]]) for evi in item['reranked_context'] if float(evi[2]) < srr_th]
        else:
            logger.info("Concatenating evidence from 'context'.")
            return [evi['id'] for evi in item['context']]
    
    def batch_retrieve(self, items: List[dict], topk: int = 1000, 
                       worker_id: int = -1, batch_size: int =1, 
                       cache_path: str = '') -> List[dict]:
        
        worker_id = 'Solo' if worker_id == -1 else worker_id
        
        logger.info(f"cache_path: {cache_path}")
        
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")
            data_list = load_pickle(cache_path)
            logger.info('Done.')
            logger.info(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
        else:
            data_list = []
            
        cached_task = len(data_list)
        total_task = len(items)
        
        assert cached_task % batch_size == 0, f"Number of cached data, {cached_task}, is not divisable by batch_size: {batch_size}."
            
        for b_start in tqdm(range(0, total_task, batch_size), total=math.ceil(total_task/batch_size), desc=f">>Worker {worker_id}: Retrieving for every query..."):
            if b_start < cached_task:
                continue
                                
            batch_items = items[b_start : b_start + batch_size]
            batch_q = [item["claim"] for item in batch_items]
            batch_context_list = self.batch_search(batch_q, topk=topk)
            
            for i in range(len(batch_items)):
                batch_items[i]["context"] = batch_context_list[i]
                
            data_list.extend(batch_items)
            
            if cache_path != '' and b_start % max(1, int(10000 / (batch_size / 2))) == 0:
                logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                dump_pickle(data_list, cache_path)
                logger.info(f">>Worker {worker_id}: Done.")
                
        if cache_path != '':
            logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
            dump_pickle(data_list, cache_path)
            logger.info(f">>Worker {worker_id}: All Done.")
        else:
            logger.info("WARNING: not saving pkl because no cache path is provided.")
            logger.info(f">>Worker {worker_id}: All Done.")
        return data_list
    
    def multi_hop_batch_retrieve(self, items: List[dict], 
                                topk: int = 1000,
                                singleHopNumbers: int = 100,
                                worker_id: int = -1, 
                                cache_path: str = '',
                                wiki_line_dict: dict=None) -> List[dict]:
        
        worker_id = 'Solo' if worker_id == -1 else worker_id
        
        logger.info(f"cache_path: {cache_path}")
        
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")
            data_list = load_pickle(cache_path)
            logger.info('Done.')
            logger.info(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
        else:
            data_list = []
            
        cached_task = len(data_list)
        total_task = len(items)

        logger.info(f"cached_task: {cached_task}")
        logger.info(f"total_task: {total_task}")

        for i in tqdm(range(total_task), total=total_task, desc=f">>Worker {worker_id}: Retrieving for every insufficient evience sentence..."):
            if i < cached_task:
                continue

            insuf_sent_ids = self.get_insuf_sent_ids(items[i], 
                                                     srr_th=args.srr_th, 
                                                     sf_th=args.sf_th)
            
            logger.info(f"len(insuf_sent_ids): {len(insuf_sent_ids)}")
            insuf_sent_ids = insuf_sent_ids[:singleHopNumbers]
            logger.info(f"len(insuf_sent_ids): {len(insuf_sent_ids)}")

            items[i]['multihop_context'] = []
            if len(insuf_sent_ids) != 0:
                lines = []
                for sentid in insuf_sent_ids:
                    evi_text = get_sentence_by_id(sentid, wiki_line_dict)
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

                if lines == []:
                    logger.info(f"WARNING: No evidence is available.")
                    continue


                logger.debug(f"lines: {lines}")

                batch_q = [" -- ".join([items[i]["claim"], evi[1]]) for evi in lines]
                
                logger.debug(f"batch_q: {batch_q}")              
                batch_context_list = self.batch_search(batch_q, topk=topk)

                sorted_batch_context_list = sorted([ctx for ctx_list in batch_context_list for ctx in ctx_list], 
                                                    key=lambda ctx: float(ctx['score']), 
                                                    reverse=True)
                
                logger.info(f"len(sorted_batch_context_list): {len(sorted_batch_context_list)}")
                 
                viewed_sent_ids = set(insuf_sent_ids)
                for j in range(len(sorted_batch_context_list)):
                    ctx = sorted_batch_context_list[j]
                    if ctx['id'] not in viewed_sent_ids:
                        viewed_sent_ids.add(ctx['id'])
                        items[i]['multihop_context'].append(ctx)

                logger.info(f"len(items[i]['multihop_context']): {len(items[i]['multihop_context'])}")
                items[i]['multihop_context'] = items[i]['multihop_context'][:topk]
                logger.info(f"len(items[i]['multihop_context']): {len(items[i]['multihop_context'])}")
                
            data_list.append(items[i])
            
            if cache_path != '' and i % 2000 == 0:
                logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                dump_pickle(data_list, cache_path)
                logger.info(f">>Worker {worker_id}: Done.")
                
        if cache_path != '':
            logger.info(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
            dump_pickle(data_list, cache_path)
            logger.info(f">>Worker {worker_id}: All Done.")
        else:
            logger.info("WARNING: not saving pkl because no cache path is provided.")
            logger.info(f">>Worker {worker_id}: All Done.")
        return data_list
    
    def batch_search(self, batch_q: List[str], topk: int = 100) -> List[List[dict]]:
        logger.info("Encoding the query batch.")            
        q_embeds = self.encode_batch(batch_q).contiguous().numpy()
        if 'HNSW' in str(type(self.index)).upper():
            q_embeds = self.convert_hnsw_query(q_embeds)
        logger.info("Done.")

        distances, indexes = self.index.search(q_embeds, topk)
        distances = distances.flat
        indexes = indexes.flat

        batch_hits = [DenseSearchResult(self.docids[idx], score)
                           for score, idx in zip(distances, indexes) if idx != -1]

        batch_context_list = []
        for i in tqdm(range(0, len(batch_hits), topk), desc='Searching for the batch.'):
            context_list = []
            hits = batch_hits[i : i + topk]
            for j in range(min(len(hits), topk)):
                context = {}
                context["id"] = str(hits[j].docid)
                context["score"] = str(hits[j].score)
                context_list.append(context)
            batch_context_list.append(context_list)

        return batch_context_list

    def convert_hnsw_query(self, query_vectors):
        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        return query_nhsw_vectors

def main(args, searching_dir):
    logger.info("Preparing model...")
    t0 = time.time()
    if args.single_encoder:
        model = SingleEncoderRetriever(encoder_dir=args.ctx_encoder_name,
                                       n_classes=3)
    else:
        model = BiEncoderRetriever(query_encoder_dir=args.query_encoder_name,
                                    ctx_encoder_dir=args.ctx_encoder_name,
                                    n_classes=3,
                                    share_encoder=args.shared_encoder)

    logger.info(f"type(model): {type(model)}")
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    model, _ = load_saved_model(model, args.checkpoint_path)
    logger.info("Done preparing model: {}: {:.2f}s".format(0, (time.time() - t0)))
   
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device %s n_gpu %d ", device, n_gpu)

    data_path = args.data_path
    logger.info(f"Loading input data from: {data_path}")
    if data_path.endswith('.pkl'):
        data_list = load_pickle(data_path)
    elif data_path.endswith('.jsonl'):
        data_list = read_jsonl(data_path)
    else:
        raise Exception(f"Unknown data format: {data_path}")
    if args.debug:
        data_list = data_list[:10]
    logger.info(f"len(data_list): {len(data_list)}")

    data_name = get_file_name(data_path)
    if args.multi_hop_dense_retrieval:
        output_file_name = f'{data_name}_single{args.singleHopNumbers}_dsrmTop{args.topk}.pkl'
    else:
        output_file_name = f'{data_name}_dsrsTop{args.topk}.pkl'
    if args.debug:
        output_file_name = "DEBUG_" + output_file_name   
    output_path = os.path.join(searching_dir, output_file_name)
    logger.info(f"output_path: {output_path}")

    if args.multi_hop_dense_retrieval:
        logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
        wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    topk = args.topk
    singleHopNumbers = args.singleHopNumbers
    batch_size = args.query_batch_size

    logger.info(f"batch_size: {batch_size}")

    file_size = len(data_list)
    num_workers = min(mp.cpu_count(), args.max_num_process)
    num_workers = num_workers if file_size // num_workers > 1 else 1

    if num_workers > 1:
        chunk_size = math.ceil(file_size / num_workers)
        chunks = split_list(data_list, chunk_num=num_workers)

        logger.info(f"num_workers: {num_workers}")
        logger.info(f"file_size: {file_size}")
        logger.info(f"chunk_size: {chunk_size}")
        logger.info(f"number of chunks: {len(chunks)}")

        results = []
        pool = mp.Pool(num_workers)
        for i in range(len(chunks)):  
            retriever = Searcher(model, 
                                tokenizer_name_or_path=args.ctx_encoder_name, 
                                sentence_corpus_path=args.sentence_corpus_path,
                                index_dir=args.index_dir,
                                index_type=args.index_type,
                                query_encoder_device=device,
                                index_in_gpu=args.index_in_gpu)
            
            cache_path = os.path.join(searching_dir, f"{data_name}_search_result_cache_chunk-{i}.pkl")
            
            if args.multi_hop_dense_retrieval:
                proc = pool.apply_async(retriever.multi_hop_batch_retrieve, 
                                        (chunks[i],), 
                                        dict(topk = topk,
                                             singleHopNumbers = singleHopNumbers,
                                             worker_id = i, 
                                             wiki_line_dict = wiki_line_dict,
                                             cache_path = cache_path))
            else:
                proc = pool.apply_async(retriever.batch_retrieve, 
                                        (chunks[i],), 
                                        dict(topk = topk,
                                             worker_id = i, 
                                             batch_size = batch_size,
                                             cache_path = cache_path))
            results.append(proc)
            
        pool.close()
        pool.join()

        retrieved_items = merge_mp_results(results)
    else:
        logger.info("Initializing Searcher object...")
        t0 = time.time()
        retriever = Searcher(model, 
                            tokenizer_name_or_path=args.ctx_encoder_name, 
                            sentence_corpus_path=args.sentence_corpus_path,
                            index_dir=args.index_dir,
                            index_type=args.index_type,
                            query_encoder_device=device,
                            index_in_gpu=args.index_in_gpu)
        logger.info("Done initializing Searcher object: {}: {:.2f}s".format(0, (time.time() - t0)))
        
        if args.cache_searching_result:
            cache_path = output_path.replace('.jsonl', '.pkl') if output_path.endswith('.jsonl') else output_path
        else:
            cache_path=''

        logger.info("\n#################################################################")
        logger.info("Start doing sentence retrival...")
        t0 = time.time()
        if args.multi_hop_dense_retrieval:
            retrieved_items = retriever.multi_hop_batch_retrieve(data_list, 
                                                                topk = topk,
                                                                singleHopNumbers = singleHopNumbers,
                                                                wiki_line_dict = wiki_line_dict,
                                                                cache_path = cache_path)
        else:
            retrieved_items = retriever.batch_retrieve(data_list, 
                                                    topk = topk, 
                                                    batch_size = batch_size,
                                                    cache_path = cache_path)
            
        logger.info("Done doing sentence retrival: {}: {:.2f}s".format(0, (time.time() - t0)))
        
    logger.info(f"len(retrieved_items): {len(retrieved_items)}")
    
    logger.info(f"Saving processed data to: {output_path}")
    t0 = time.time()
    if output_path.endswith('.pkl'):
        dump_pickle(retrieved_items, output_path)
        logger.info("Done.")
    elif output_path.endswith('.jsonl'):
        save_jsonl(retrieved_items, output_path)
    else:
        raise Exception("Illigal output path: {output_path}")
    logger.info("Done saving processed data: {}: {:.2f}s".format(0, (time.time() - t0)))
    
    return


if __name__ == '__main__':
    if args.multi_hop_dense_retrieval:
        log_file = 'multi_hop_searcher'
    else:
        log_file = 'single_hop_searcher'

    searching_dir = os.path.join(get_file_dir(args.checkpoint_path), 'search_results')
    if not os.path.exists(searching_dir):
            make_directory(searching_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(searching_dir, f"{log_file}.log"))

    logger.info("Start main")
    t_start = time.time()

    main(args, searching_dir)

    logger.info("All done: {}: {:.2f}s".format(0, (time.time() - t_start)))
