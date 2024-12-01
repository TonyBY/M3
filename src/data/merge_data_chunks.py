import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import read_jsonl, load_pickle, dump_pickle, read_npy, save_npy
from tqdm import tqdm
import numpy as np

import os

import argparse
import logging

logger = logging.getLogger()

from src.utils.args import ArgumentGroup
from src.utils.args import prepare_logger

parser = argparse.ArgumentParser(__doc__)

data_merge_g = ArgumentGroup(parser, "data merge", "data merge options.")
data_merge_g.add_arg("npy",                              bool,          True,  
                "If data files to merge are in .npy format.")
data_merge_g.add_arg("number_of_chunks",                 int,           16,
                "Number of data files to merge.")
data_merge_g.add_arg("data_chunk_dir",                    str,          "",
                "Directory where the data chunk files are saved.")
data_merge_g.add_arg("data_file_name_template",           str,          "",
                "Pattern of the data chunk file names, e.g., vectors_CHUNKNUM.npy")

def merge_data_chunks(data_chunk_path_template: str = None, 
                      number_of_chunks: int = None,
                      file_type: str = 'pkl'):
    
    all_data = []
    for i in tqdm(range(number_of_chunks), desc='Merging data chunks...'):
        cache_path = data_chunk_path_template.replace('CHUNK_NUMBER_PLACE_HOLDER', str(i))
        logger.info(f"Loading cached data_list from: {cache_path}")
        if file_type == 'pkl':
            data_list = load_pickle(cache_path)
        elif file_type == 'jsonl':
            data_list = read_jsonl(cache_path)
        else:
            raise Exception(f"Unsupported file type: {file_type}")
        logger.info('Done.')
        logger.info(f"len(data_list): {len(data_list)}")

        all_data.extend(data_list)
        logger.info(f"len(all_data): {len(all_data)}")
    
    output_path = data_chunk_path_template.replace("CHUNK_NUMBER_PLACE_HOLDER", f"all_{len(all_data)}")
    logger.info(f"Saving all_data of length {len(all_data)} to: {output_path}")
    dump_pickle(all_data, output_path)
    logger.info(f"Done.")


def merge_data_chunks_npy(data_chunk_path_template: str = None, 
                      number_of_chunks: int = None):
    
    embedding_list = []
    for i in tqdm(range(number_of_chunks), desc='Merging data chunks...'):
        cache_path = data_chunk_path_template.replace('CHUNK_NUMBER_PLACE_HOLDER', str(i))
        logger.info(f"Loading cached embeddings from: {cache_path}")
        
        embeddings = read_npy(cache_path)
        
        logger.info('Done.')
        logger.info(f"embeddings.shape: {embeddings.shape}")

        embedding_list.append(embeddings)
        logger.info(f"len(embedding_list): {len(embedding_list)}")
    
    embedding_concat = np.vstack(embedding_list)
    
    output_path = data_chunk_path_template.replace("_CHUNK_NUMBER_PLACE_HOLDER", f"")
    logger.info(f"Saving embedding_concat of shape {embedding_concat.shape} to: {output_path}")
    save_npy(embedding_concat, output_path)
    logger.info(f"Done.")


def main(args):
    data_chunk_path_template = os.path.join(args.data_chunk_dir, args.data_file_name_template)
    file_type = data_chunk_path_template.split('.')[-1]
    logger.info(f"file_type: {file_type}")

    if args.npy:
        merge_data_chunks_npy(data_chunk_path_template = data_chunk_path_template,
                  number_of_chunks = args.number_of_chunks)
    else:
        merge_data_chunks(data_chunk_path_template = data_chunk_path_template, 
                  number_of_chunks = args.number_of_chunks,
                  file_type = file_type)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    prepare_logger(logger, debug=False, save_to_file=os.path.join(args.data_chunk_dir, "merge_chunks.log"))
    logger.info(f"All Done.")
    