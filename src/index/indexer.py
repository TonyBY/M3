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

from typing import Union
from tqdm import tqdm
import faiss
from numpy import ndarray
import numpy as np
import logging

import torch
from torch import Tensor as T

import transformers

from src.models.encoder import SentenceEncoder
from src.models.joint_retrievers import BiEncoderRetriever, SingleEncoderRetriever
from src.utils.model_utils import load_saved_model
from src.utils.data_utils import make_directory, read_jsonl, load_pickle, dump_pickle
from src.utils.config import parser
from src.utils.args import prepare_logger

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

torch.cuda.empty_cache()

class Indexer(SentenceEncoder):
    def __init__(self, 
                model,
                tokenizer_name_or_path:str=None, 
                max_length:int=512,
                sentence_type:str="ctx",
                sentence_corpus_path:str=None,
                index_dir:str=None,
                device=None,):
        
        super().__init__(model,
                         tokenizer_name_or_path=tokenizer_name_or_path, 
                         max_length=max_length,
                         sentence_type=sentence_type)
        
        logger.info(f"type(self.encoder): {type(self.encoder)}")

        self.sentence_corpus_path = sentence_corpus_path
        self.sentence_embedding_path = os.path.join(index_dir, 'vectors.npy') if index_dir else ""
        self.index_dir = index_dir
        self.device = device
        self.encoder = self.encoder.to(self.device)
        
        if index_dir and not os.path.exists(self.index_dir):
            make_directory(self.index_dir)

        self.sentence_corpus = None

        self.doc_id_path = os.path.join(self.index_dir, 'docid') if self.index_dir else None
        
    def encode(self, return_numpy: bool = True,
                    normalize_to_unit: bool = False,
                    keepdim: bool = False,
                    batch_size: int = 64,
                    max_length: int = 512,
                    shard_id: int = 0,
                    shard_number: int = 1,
                    cache_path: str = '') -> Union[ndarray, T]:

        if cache_path == '':
            cache_path = os.path.join(self.index_dir, f'embedding_cache_{shard_id}.pkl')
        
        if not self.sentence_corpus:
            logger.info("Loading sentences from %s ..." % (self.sentence_corpus_path))
            if self.sentence_corpus_path.endswith('.pkl'):
                self.sentence_corpus = load_pickle(self.sentence_corpus_path)
            else:
                self.sentence_corpus = read_jsonl(self.sentence_corpus_path)
            logger.info(f"len(sentence_corpus): {len(self.sentence_corpus)}")

        logger.info(f"batch_size: {batch_size}")
        shard_size = len(self.sentence_corpus) // shard_number
        logger.info(f"shard_id: {shard_id}")
        logger.info(f"shard_number: {shard_number}")
        logger.info(f"shard_size: {shard_size}")
        logger.info(f"Updated len(self.sentence_corpus): {len(self.sentence_corpus)}")

        if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
            logger.info(f"Loading cached embedding_list from: {cache_path}")
            embedding_list = load_pickle(cache_path)
            logger.info('Done.')
            logger.info(f"len(embedding_list): {len(embedding_list)}")
        else:
            embedding_list = []
            
        if embedding_list != []:
            assert len(embedding_list[-1]) % batch_size == 0, f"Number of cached batches, {cached_batch}, is not divisable by batch_size: {batch_size}."

        cached_batch = len(embedding_list)
        total_batch = len(self.sentence_corpus) // batch_size + (1 if len(self.sentence_corpus) % batch_size > 0 else 0)

        with torch.no_grad():
            for batch_id in tqdm(range(total_batch), desc='Encoding sentence corpos...'):
                if batch_id < cached_batch:
                    continue

                embeddings = self.encode_batch(self.sentence_corpus[batch_id*batch_size:(batch_id+1)*batch_size])
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings)

                if batch_id % 50 == 0:
                    logger.info(f"Saving embedding_list of length {len(embedding_list)} to: {cache_path}")
                    dump_pickle(embedding_list, cache_path)
                    logger.info(f"Done.")

        embeddings = torch.cat(embedding_list, 0)
        
        if return_numpy and not isinstance(embeddings, ndarray):
            vectors_file = os.path.join(self.index_dir, f'vectors_{shard_id}.npy')
            logger.info(f"Saving vecotors to: {vectors_file}")
            np.save(vectors_file, embeddings)
            logger.info(f"Finish saving vecotors to: {vectors_file}.")
            return embeddings.numpy()
        
        return embeddings
    
    def get_doc_ids(self):
        logger.info("Loading sentences from %s ..." % (self.sentence_corpus_path))
        if self.sentence_corpus_path.endswith('.pkl'):
            self.sentence_corpus = load_pickle(self.sentence_corpus_path)
        else:
            self.sentence_corpus = read_jsonl(self.sentence_corpus_path)
        logger.info(f"len(sentence_corpus): {len(self.sentence_corpus)}")

        with open(self.doc_id_path, 'a') as f:
            for sent in tqdm(self.sentence_corpus, desc='Generating docid file...'):
                doc_id = sent['id']
                f.write(f'{doc_id}\n')
        
    def build_index(self,
                    encoding_batch_size: int = 64,
                    indexing_batch_size: int = 50000,
                    index_type: str = 'HNSWFlat',
                    sentence_embedding_path:str='',
                    index_dim: int=768):

        if sentence_embedding_path != '':
            logger.info(f"Loading pre-trained vectors from: {sentence_embedding_path}")
            xb = np.load(sentence_embedding_path).astype('float32')
        elif os.path.exists(self.sentence_embedding_path):
            logger.info(f"Loading pre-trained vectors from: {self.sentence_embedding_path}")
            xb = np.load(self.sentence_embedding_path).astype('float32')
        else:
            logger.info("Encoding embeddings for sentences...")
            xb = self.encode(batch_size=encoding_batch_size, 
                             normalize_to_unit=False,
                             return_numpy=True)
            
        logger.info(f"type(xb): {type(xb)}") 
        logger.info(f"xb.shape: {xb.shape}")
        
        output_path = os.path.join(self.index_dir, f'{index_type}_index')
        
        logger.info("Building index...")
        if index_type == 'IndexFlatIP':
            if os.path.exists(output_path):
                logger.info(f"Loading pretrained index from: {output_path}")
                index= faiss.read_index(output_path)
                logger.info("Done.")
                logger.info(f"index.ntotal: {index.ntotal}")
                logger.info(f"index.is_trained: {index.is_trained}")
            else:

                dim = index_dim
                index = faiss.IndexFlatIP(dim)

                logger.info("Start adding vectors to trained index.")
                index.add(xb)
                logger.info('Done.')
                logger.info(f"index.ntotal: {index.ntotal}")

                logger.info(f"Saving index to: {output_path}")
                faiss.write_index(index, output_path)
                logger.info("All Done.")
        elif index_type == 'IVF16384_HNSW32':
            if os.path.exists(output_path):
                logger.info(f"Loading pretrained index from: {output_path}")
                index= faiss.read_index(output_path)
                logger.info("Done.")
                logger.info(f"index.ntotal: {index.ntotal}")
                logger.info(f"index.is_trained: {index.is_trained}")
            else:
                dim, measure = index_dim, faiss.METRIC_L2
                param = "PCA64,IVF16384_HNSW32,Flat"
                index_hnsw = faiss.index_factory(dim, param, measure)

                # Doing training in GPUs, and everything else in cpu.
                index_ivf = faiss.extract_index_ivf(index_hnsw)
                    
                clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d)) if self.device == 'cuda' else faiss.IndexFlatL2(index_ivf.d)
                index_ivf.clustering_index = clustering_index

                logger.info("Start train clusters for ivf_hnsw index.")
                index_hnsw.train(xb)
                index = index_hnsw
                logger.info("Done.")

                logger.info("Start adding vectors to trained index.")
                index.add(xb)
                logger.info('Done.')
                logger.info(f"index.ntotal: {index.ntotal}")

                logger.info(f"Saving index to: {output_path}")
                faiss.write_index(index, output_path)
                logger.info("All Done.")

        elif index_type == 'HNSWFlat':
            # Geting L2 space phi for  DotProduct(InnerProduct) measurement.
            phi_caching_path = os.path.join(self.index_dir, 'phi.pkl')
            
            if os.path.exists(phi_caching_path):
                logger.info(f"Loading chached L2 space phi from: {phi_caching_path}")
                phi = load_pickle(phi_caching_path)
                logger.info("Done.")
            else:
                phi = 0
                for i, vec in tqdm(enumerate(xb), desc='Calculating L2 space phi.'):
                    norms = (vec ** 2).sum()
                    phi = max(phi, norms)
                logger.info("Done.")
                logger.info(f"Saving L2 space phi to: {phi_caching_path}")
                dump_pickle(phi, phi_caching_path)
                logger.info("Done.")

                logger.info(f'HNSWF DotProduct -> L2 space phi={phi}')

            # Starting from cached index if available.
            if os.path.exists(output_path):
                logger.info(f"Loading cached index from: {output_path}")
                index = faiss.read_index(output_path)
                logger.info(f"Continuing from: {index.ntotal}/{len(xb)}")
                xb = xb[index.ntotal:]
                logger.info(f"Remaining vectors to add: {len(xb)}")
            else:
                d = index_dim
                index = faiss.IndexHNSWFlat(d + 1, 512)
                index.hnsw.efSearch = 128
                index.hnsw.efConstruction = 200

            # Normalizing and adding new vectors to the index.
            data = xb
            buffer_size = indexing_batch_size
            n = len(data)
            for i in tqdm(range(0, n, buffer_size), desc=f'Normalizing and adding data into index with batchsize of: {buffer_size}'):
                vectors = [np.reshape(t, (1, -1)) for t in data[i:i + buffer_size]]
                norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
                aux_dims = [np.sqrt(phi - norm) for norm in norms]
                hnsw_vectors = [np.hstack((doc_vector, aux_dims[idx].reshape(-1, 1))) for idx, doc_vector in enumerate(vectors)]
                hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
                logger.info(f"hnsw_vectors.shape: {hnsw_vectors.shape}")
                index.add(hnsw_vectors)

                # Caching the intermediate results.
                if i % (10 * buffer_size) == 0:
                    logger.info(f"Caching index to: {output_path}")
                    faiss.write_index(index, output_path)
                    logger.info("Done.")

            # Output the new index.
            logger.info(f"Saving index to: {output_path}")
            faiss.write_index(index, output_path)
            logger.info("All Done.")
        else:
            raise Exception(f"Unknow index type: {index_type}. Only supports: {'IVF16384_HNSW32' and 'HNSWFlat'} for now.")

if __name__ == '__main__': 
    if not os.path.exists(args.index_dir):
        # Sometimes Slurm cannot find the dir when doing os.path.exists check, but raise FileExistsError when doing make_directory.
        try:
            make_directory(args.index_dir)
        except FileExistsError:
            pass

    if args.encode_only:
        prepare_logger(logger, debug=False, save_to_file=os.path.join(args.index_dir, "encoder.log"))
    else:
        prepare_logger(logger, debug=False, save_to_file=os.path.join(args.index_dir, "indexer.log"))

    if args.single_encoder:
        model = SingleEncoderRetriever(encoder_dir=args.ctx_encoder_name,
                                       n_classes=3)
    else:
        model = BiEncoderRetriever(query_encoder_dir=args.query_encoder_name,
                                    ctx_encoder_dir=args.ctx_encoder_name,
                                    n_classes=3,
                                    share_encoder=args.shared_encoder)

    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    model, _ = load_saved_model(model, args.checkpoint_path)

   
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device %s n_gpu %d ", device, n_gpu)
    
    indexer = Indexer(model, 
                      tokenizer_name_or_path=args.ctx_encoder_name, 
                      sentence_corpus_path=args.sentence_corpus_path,
                      index_dir=args.index_dir,
                      device=device)

    if args.encode_only:
        logger.info("Only encoding corpus.")
        indexer.encode(batch_size=args.encoding_batch_size, 
                       normalize_to_unit=False,
                       return_numpy=True,
                       shard_id=args.shard_id)
        logger.info("All Done.")
    else:
        logger.info(f"Building index for the encoded corpus at {args.sentence_embedding_path}.")

        if not os.path.exists(indexer.doc_id_path):
            indexer.get_doc_ids()

        indexer.build_index(encoding_batch_size=args.encoding_batch_size,
                        indexing_batch_size=args.indexing_batch_size,
                        index_type=args.index_type,
                        sentence_embedding_path=args.sentence_embedding_path,
                        index_dim=args.index_dim)
        logger.info("All Done.")
