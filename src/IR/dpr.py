from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os

from src.models.encoder import DprQueryBatchEncoder
from src.utils.data_utils import load_docids, get_file_dir, make_directory, read_jsonl, save_jsonl, load_pickle, dump_pickle, split_list, merge_mp_results
from src.IR.base import Retriever

from typing import List
from tqdm import tqdm
from dataclasses import dataclass

import multiprocessing as mp
import torch
import faiss
import math
import numpy as np

@dataclass
class DenseSearchResult:
    docid: str
    score: float


class DPR_Retriever(Retriever):
    def __init__(self,
                 index_path: str = None,
                 docids_path: str = None,
                 query_encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
                 index_in_gpu: bool = False,
                 query_encoder_device: str = 'cpu',
                 max_length: int = 70):
        
        print(f"Loading docids from: {docids_path}")
        self.docids = load_docids(docids_path)
        print("Done.")
        
        print(f"Loading index from: {index_path}")
        self.index = faiss.read_index(index_path)
        print("Done.")
        
        print(f"Initializing DprQueryBatchEncoder with model: {query_encoder_name}; at device: {query_encoder_device}: with max_length: {max_length}.")
        self.dpr_encoder = DprQueryBatchEncoder(encoder_dir=query_encoder_name, device=query_encoder_device, max_length=max_length)
        print("Done.")

        if index_in_gpu and self.encoder.device.type == 'cuda':
            if 'HNSW' in str(type(self.index)).upper():
                print(f"Faiss HNSW index does not support running on gpu.")
                print(f"Using cpu for searching.")
            else:
                # Note, faiss HNSW index does not support running on gpu. 
                # It will just copy the index to another CPU index (the default behavior for indexes it does not recognize).
                res = faiss.StandardGpuResources()
                print("Loading index to gpu.")
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index) # Use the gpu of index 0.
                print("Done.")
        else:
            print(f"Using cpu for searching.")

    def batch_retrieve(self, items: List[dict], topk: int = 1000, 
                       worker_id: int = -1, batch_size: int =1, 
                       cache_path: str = '') -> List[dict]:
        
        worker_id = 'Solo' if worker_id == -1 else worker_id
        
        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            print(f">>Worker {worker_id}: Loading cached data_list from: {cache_path}")
            data_list = load_pickle(cache_path)
            print('Done.')
            print(f">>Worker {worker_id}: len(data_list): {len(data_list)}")
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
            
            if b_start % 1000 == 0:
                print(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
                dump_pickle(data_list, cache_path)
                print(f">>Worker {worker_id}: Done.")
                
        print(f">>Worker {worker_id}: Saving data_list of length {len(data_list)} to: {cache_path}")
        dump_pickle(data_list, cache_path)
        print(f">>Worker {worker_id}: All Done.")
        return data_list
    
    def batch_search(self, batch_q: List[str], topk: int = 1000) -> List[List[dict]]:            
        q_embeds = self.dpr_encoder.encode_batch(batch_q)

        if 'HNSW' in str(type(self.index)).upper():
            q_embeds = self.convert_hnsw_query(q_embeds)
            
        distances, indexes = self.index.search(q_embeds, topk)
        distances = distances.flat
        indexes = indexes.flat

        batch_hits = [DenseSearchResult(self.docids[idx], score)
                           for score, idx in zip(distances, indexes) if idx != -1]

        batch_context_list = []
        for i in range(0, len(batch_hits), topk):
            context_list = []
            hits = batch_hits[i : i + topk]
            for j in range(topk):
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


def main(data_path: str = '', 
         index_path: str = '', 
         docids_path : str = '', 
         output_path: str = '', 
         max_num_process: int = 1, 
         topk: int = 1000,
         batch_size: int = 1,
         query_encoder_name: str = 'facebook/dpr-question_encoder-multiset-base',
         index_in_gpu: bool = False,
         query_encoder_device: str = 'cpu',
         max_length: int = 70):
    
    print(f"type(output_path): {type(output_path)}")
    print(f"output_path: {output_path}")
    output_dir = get_file_dir(output_path) + '/cache'
    make_directory(output_dir)

    print(f"{output_dir}: {output_dir}")

    print(f"Loading queries from: {data_path}")
    data_list = read_jsonl(data_path)
    print(f"len(data_list): {len(data_list)}")

    file_size = len(data_list)
    num_workers = min(mp.cpu_count(), max_num_process)
    num_workers = num_workers if file_size // num_workers > 1 else 1

    if num_workers > 1:
        chunk_size = math.ceil(file_size / num_workers)
        chunks = split_list(data_list, chunk_num=num_workers)

        print(f"num_workers: {num_workers}")
        print(f"file_size: {file_size}")
        print(f"chunk_size: {chunk_size}")
        print(f"number of chunks: {len(chunks)}")

        results = []
        pool = mp.Pool(num_workers)
        for i in range(len(chunks)):  
            retriever = DPR_Retriever(index_path = index_path, 
                                  docids_path = docids_path,
                                  index_in_gpu = index_in_gpu, 
                                  query_encoder_device = query_encoder_device, 
                                  query_encoder_name = query_encoder_name, 
                                  max_length = max_length)
            
            cache_path = f"{output_dir}/{i}.pkl" 
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
        # Serial method:  
        retriever = DPR_Retriever(index_path = index_path, 
                                  docids_path = docids_path,
                                  index_in_gpu = index_in_gpu, 
                                  query_encoder_device = query_encoder_device, 
                                  query_encoder_name = query_encoder_name, 
                                  max_length = max_length)
        
        cache_path = output_path.replace('.jsonl', '.pkl') if output_path.endswith('.jsonl') else output_path
        retrieved_items = retriever.batch_retrieve(data_list, 
                                                   topk = topk, 
                                                   batch_size = batch_size,
                                                   cache_path = cache_path)
        
    print(f"len(retrieved_items): {len(retrieved_items)}")
    
    if output_path.endswith('.jsonl'):
        print(f"Saving processed data to: {output_path}")
        save_jsonl(retrieved_items, output_path)
        print("Done.")

    return

    
if __name__ == "__main__":
    import gc
    import transformers

    gc.enable()
    torch.cuda.empty_cache() 
    transformers.logging.set_verbosity_error()
    
    data_path = 'M3/data/FEVER_1/shared_task_dev.jsonl'
    
    index_path = 'M3/data/pyserini/index/zero_shot_single_dpr_noNER/index'
    docids_path = 'M3/data/pyserini/index/zero_shot_single_dpr_noNER/docid'
    output_path = 'M3/data/results/IR/dpr/zero_single/flatIndex/retrieval_results.jsonl'
    query_encoder_name = 'facebook/dpr-question_encoder-single-nq-base'
    
    max_num_process = 1
    topk = 1000
    batch_size = 64
    index_in_gpu = True
    query_encoder_device = 'cuda:0'
    max_length = 70
        
    main(data_path = data_path, 
        index_path = index_path, 
        docids_path = docids_path, 
        output_path = output_path, 
        max_num_process = max_num_process, 
        topk = topk,
        batch_size = batch_size,
        query_encoder_name = query_encoder_name,
        index_in_gpu = index_in_gpu,
        query_encoder_device = query_encoder_device,
        max_length = max_length)
