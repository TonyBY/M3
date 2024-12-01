import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import get_file_dir, get_file_name, make_directory,\
                                 load_pickle, dump_pickle, \
                                 read_lines_from_file, \
                                 process_evid

from src.models.joint_retrievers import BiEncoderRetriever, SingleEncoderRetriever
from src.utils.model_utils import load_saved_model

from src.IR.searcher import Searcher

from src.utils.config import parser
from src.utils.args import prepare_logger

import torch
from numpy import ndarray

from typing import List
from tqdm import tqdm
import faiss
import transformers
import logging

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()


def get_sent_idx_by_titles(title_list: List[str],  
                          sep: str='|#SEP#|',
                          docid_to_sent_idx_dict: dict=None,
                          sentids: List[str]=None,) -> List[int]:
    sent_idx_list = []
    if docid_to_sent_idx_dict:
        for title in title_list:
            try:
                sent_idx_list.extend(docid_to_sent_idx_dict[title])
            except KeyError:
                logger.info(f"Warning: {title} is not in the corpus.")
                continue
    else:
        for sent_idx, sentid in enumerate(sentids):
            docid = process_evid(sentid.split(sep)[0])
            if docid in title_list:
                sent_idx_list.append(sent_idx)
                logger.info(docid)
    return sent_idx_list

def build_docid_to_sent_idx_map(sentids: List[str], sep: str='|#SEP#|') -> dict:
    docid_to_sent_idx_dict = {}
    for sent_idx, sentid in tqdm(enumerate(sentids), 
                                 total=len(sentids), 
                                 desc="Building docid_to_sent_idx_map..."):
        docid = process_evid(sentid.split(sep)[0])
        if docid not in docid_to_sent_idx_dict:
            docid_to_sent_idx_dict[docid] = [sent_idx]
        else:
            docid_to_sent_idx_dict[docid].append(sent_idx)
    return docid_to_sent_idx_dict

def normalize_scores(scores: List[float]) -> List[float]:
    normalized_scores = []
    max_score = max(scores)
    min_score = min(scores)
    
    for score in scores:
        normalized_score = (score - min_score + 1e-1) / (max_score - min_score + 1e-1)
        normalized_scores.append(normalized_score)
    return normalized_scores

def get_vectors_from_faiss_index(index_path: str)-> ndarray:
    index = faiss.read_index(index_path)
    vectors = index.reconstruct_n(0, index.ntotal)
    return vectors

def get_second_hop_faiss_index(complete_corpus_vectors: ndarray=None, 
                               sent_idx_list: List[int]=None,
                               index_type: str='IndexFlatIP',
                               dim=768):
    second_hop_corpus_vectors = complete_corpus_vectors[sent_idx_list]
    if index_type == 'IndexFlatIP':
        second_hop_index = faiss.IndexFlatIP(dim)
        second_hop_index.add(second_hop_corpus_vectors)
    elif index_type == 'HNSWFlat':
        phi = 0
        for vec in tqdm(second_hop_corpus_vectors, desc='Calculating L2 space phi.'):
            norms = (vec ** 2).sum()
            phi = max(phi, norms)
        logger.info("Done.")
        logger.info(f'HNSWF DotProduct -> L2 space phi={phi}')
        second_hop_index = faiss.IndexHNSWFlat(dim + 1, 512)
        second_hop_index.hnsw.efSearch = 128
        second_hop_index.hnsw.efConstruction = 200
    else:
        raise Exception(f"Unknown index_type: {index_type}. Only support IndexFlatIP and HNSWFlat for now.")
        
    return second_hop_index

def second_hop_search(first_hop_search_results: List[dict],
                      model=None, 
                      topk: int=None, 
                      wiki_line_dict: dict=None, 
                      wiki_extra_line_dict: dict=None,
                      docid_to_sent_idx_dict: dict=None,
                      complete_corpus_vectors: ndarray =None,
                      index_type: str=None,
                      device=None,
                      index_in_gpu: bool=None,
                      sentids: List[str]=None,
                      ):
    second_hop_search_results = {}
    for item in tqdm(first_hop_search_results, 
                     total=len(first_hop_search_results),
                     desc="Start doing second hop searching..."):
        first_hop_evi = item['context'][:topk] 
        for evi in first_hop_evi:
            if evi['id'] not in wiki_line_dict: # For debugging.
                continue
            query = process_evid(wiki_line_dict[evi['id']])
            query = [{'claim': query}]

    #         print(f"evi['id']: {evi['id']}")
            linked_docs = list(set(wiki_extra_line_dict[evi['id']].split('\t')))
    #         print(f"linked_docs: {linked_docs}")
            
            second_hop_search_corpus_sentIdx = get_sent_idx_by_titles(linked_docs,  
                                                                      docid_to_sent_idx_dict=docid_to_sent_idx_dict)
            if second_hop_search_corpus_sentIdx == []:
                continue
    #         print(f"second_hop_search_corpus_sentIdx: {second_hop_search_corpus_sentIdx}")
            
            second_hop_faiss_index=get_second_hop_faiss_index(complete_corpus_vectors=complete_corpus_vectors,
                                                              sent_idx_list=second_hop_search_corpus_sentIdx,
                                                              index_type=index_type,
                                                             )
            
            seond_hop_docids = [sentids[idx] for idx in second_hop_search_corpus_sentIdx]
            retriever = Searcher(model, 
                                tokenizer_name_or_path=args.ctx_encoder_name, 
                                sentence_corpus_path=None,
                                index=second_hop_faiss_index,
                                index_type=args.index_type,
                                query_encoder_device=device,
                                index_in_gpu=args.index_in_gpu,
                                docids=seond_hop_docids)
            
            retrieved_items = retriever.batch_retrieve(query, 
                                                       topk=topk, 
                                                       batch_size=1,
                                                      )
            
            second_hop_search_result_id = str(item['id']) + '|#ID#|' + evi['id']
            second_hop_search_results[second_hop_search_result_id] = retrieved_items[0]['context']
    return second_hop_search_results


def main(args):
    if not os.path.exists(args.index_dir):
            make_directory(args.index_dir)
        
    prepare_logger(logger, debug=False, save_to_file=os.path.join(args.index_dir, "secondHop_searcher.log"))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    n_gpu = torch.cuda.device_count()
    logger.info("device %s n_gpu %d ", device, n_gpu)

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

    data_name = get_file_name(args.data_path)
    first_hop_search_results_path = os.path.join(args.index_dir, f'{data_name}_search_result.pkl')
    logger.info(f"Loading the first_hop_search_results from: {first_hop_search_results_path}")
    first_hop_search_results = load_pickle(first_hop_search_results_path)

    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    logger.info(f"Loading wiki_extra_line_dict from: {args.wiki_extra_line_dict_pkl_path}")
    wiki_extra_line_dict = load_pickle(args.wiki_extra_line_dict_pkl_path)

    doc_id_path = os.path.join(args.index_dir, 'docid') if args.index_dir else None
    sentids = read_lines_from_file(doc_id_path)

    docid_to_sent_idx_dict_dir = get_file_dir(args.docid_to_sent_idx_dict_pkl_path)
    if os.path.exists(args.docid_to_sent_idx_dict_pkl_path):
        logger.info(f"Loading docid_to_sent_idx_dict from: {args.docid_to_sent_idx_dict_pkl_path}")
        docid_to_sent_idx_dict = load_pickle(args.docid_to_sent_idx_dict_pkl_path)
    else:
        logger.info(f"Building docid_to_sent_idx_dict...")
        docid_to_sent_idx_dict = build_docid_to_sent_idx_map(sentids)
        docid_to_sent_idx_dict_pkl_path = os.path.join(docid_to_sent_idx_dict_dir, 
                                                       f"fever_docid_to_sent_idx_dict_{len(docid_to_sent_idx_dict)}.pkl")
        dump_pickle(docid_to_sent_idx_dict, docid_to_sent_idx_dict_pkl_path)

    index_path = os.path.join(args.index_dir, f'{args.index_type}_index')
    complete_corpus_vectors = get_vectors_from_faiss_index(index_path)

    second_hop_search_results = second_hop_search(first_hop_search_results, 
                                                    model=model,
                                                    topk=args.topk, 
                                                    wiki_line_dict=wiki_line_dict, 
                                                    wiki_extra_line_dict=wiki_extra_line_dict,
                                                    docid_to_sent_idx_dict=docid_to_sent_idx_dict,
                                                    complete_corpus_vectors=complete_corpus_vectors,
                                                    index_type=args.index_type,
                                                    device=device,
                                                    index_in_gpu=args.index_in_gpu,
                                                    sentids=sentids
                                                    )

    output_path = os.path.join(args.index_dir, f'second_hop_search_result.pkl')
    logger.info(f"Saving second_hop_search_results to: {output_path}")
    dump_pickle(second_hop_search_results, output_path)


if __name__ == "__main__":
    main(args)