from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from typing import List
from tqdm import tqdm

import torch
import torch.nn.functional as F

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import numpy as np

import argparse
import logging

from src.utils.data_utils import get_file_dir, get_file_name, read_jsonl, save_jsonl, load_pickle, dump_pickle, move_to_device
from src.utils.config import parser
from src.utils.args import prepare_logger

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'

logger = logging.getLogger()

def define_args(parser):
    parser.add_argument('--data_path',
                        type=str,
                        required=False,
                        default="M3/data/dpr/fever_bm25/train/all/withIterativeMultiHop/data_50negs_100.jsonl"
                        )

    parser.add_argument('--neg_size',
                        type=int,
                        required=False,
                        default=50)
            
    parser.add_argument('--device',
                        type=str,
                        required=False,
                        default='cuda')

    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        default='cross-encoder/ms-marco-MiniLM-L-12-v2')

    parser.add_argument('--threshold',
                        type=float,
                        required=False,
                        default=0.999)

    parser.add_argument('--max_len',
                        type=int,
                        required=False,
                        default=512)

def filter_misnegatives(input_data: List[dict], 
                        model_name:str, device:str='cpu', 
                        threshold:float=0.9, 
                        max_len:int=512,
                        cache_path:str=None,
                        keep_first_neg:bool=False):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info(f"labels: {model.config.id2label}")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(cache_path) and cache_path[-4:] == '.pkl':
        logger.info(f"Loading cached data_list from: {cache_path}")
        data_list = load_pickle(cache_path)
        logger.info('Done.')
        logger.info(f"len(data_list): {len(data_list)}")
    else:
        data_list = []
    
    cached_task = len(data_list)
    total_task = len(input_data)
    logger.info(f"Resuming from: {cached_task}/{total_task}.")
    
    for i in tqdm(range(total_task), total=total_task, 
                desc="Cleaning sampled hard-negs ..."):
        
        if i < cached_task:
            continue
                    
        data = input_data[i]
        logger.info("###############################")
        claim = data['question']
        logger.info(f"claim {i}: {claim}")
        anno = data['answers']
        logger.info(f"anno: {anno}")
        logger.info('-------------------------------')

        correct_negs = []
        for idx, neg in enumerate(data['hard_negative_ctxs']):
            if keep_first_neg and idx==0:
                continue
            neg_ctx = neg['title'] + ' . ' + neg['text']
            text_pair_codes = tokenizer.encode_plus(claim.strip(),
                                                    text_pair=neg_ctx.strip(),
                                                    max_length=max_len,
                                                    truncation=True,
                                                    padding='max_length', 
                                                    return_tensors="pt")
            text_pair_codes = move_to_device(dict(text_pair_codes), device)
            with torch.no_grad():
                logits = model(**text_pair_codes).logits
            
            if logits.shape[1] == 1: 
                conf = torch.sigmoid(logits).cpu().numpy()
                label = model.config.id2label[0]
            else:
                probs = F.softmax(logits, dim=1).cpu().numpy()
                conf, predicted_class_id = np.max(probs), logits.argmax().item()
                label = model.config.id2label[predicted_class_id]

            if conf > threshold:
                logger.info(f"neg: {neg}")
                logger.info(f"conf: {conf}")
                logger.info(f"label: {label}")
                logger.info('=============================')
            else:
                correct_negs.append(neg)
                
        data['hard_negative_ctxs'] = correct_negs
        data_list.append(data)
        if len(data['hard_negative_ctxs']) == 0:
            print(f"Warning: claim{i}: '{claim}', has zero cleaned negatives.")
        
        if i % 50 == 0:
            logger.info(f"Saving data_list of length {len(data_list)} to: {cache_path}")
            dump_pickle(data_list, cache_path)
            logger.info(f"Done.")

    logger.info(f"Saving data_list of length {len(data_list)} to: {cache_path}")
    dump_pickle(data_list, cache_path)
    logger.info(f"Done.")

    return data_list
        
def main(args):
    data_path = args.data_path
    neg_size=args.neg_size
    device = args.device
    model_name = args.model_name
    threshold = args.threshold
    max_len=args.max_len
    
    output_dir = get_file_dir(data_path)
    data_version_name = f"{get_file_name(data_path)}_neg-{neg_size}_cleanedNegs_model-{model_name.replace('/', '_')}_threshold-{threshold}"
    prepare_logger(logger, debug=False, save_to_file=os.path.join(output_dir, f"clean_neg_{data_version_name}.log"))

    logger.info(f"Loading data from: {data_path}")
    if data_path.endswith(".pkl"):
        data_list = load_pickle(data_path)
    else:
        data_list = read_jsonl(data_path)

    cache_path = os.path.join(output_dir, f"{data_version_name}_size-{len(data_list)}_cache.pkl")
    output_path = os.path.join(output_dir, f"{data_version_name}_size-{len(data_list)}.jsonl")

    data_list = filter_misnegatives(data_list, 
                                    model_name, 
                                    device=device, 
                                    threshold=threshold, 
                                    max_len=max_len, 
                                    cache_path=cache_path)
    
    logger.info(f"saving processed data to {output_path}")
    save_jsonl(data_list, output_path)
    
    save_jsonl(data_list[:100], os.path.join(output_dir, f"{data_version_name}_size-100.jsonl"))
    
    logger.info(f"All Done.")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean sampled hard-negatives with pre-trained sentene ranker.')
    define_args(parser)
    args = parser.parse_args()    
    main(args)
