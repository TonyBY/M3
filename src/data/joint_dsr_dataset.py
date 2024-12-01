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

from src.utils.data_utils import load_pickle

from torch.utils.data import Dataset, WeightedRandomSampler
from torch import Tensor as T
import torch

import collections

import json
import random
import logging
from tqdm import tqdm
from typing import List

logger = logging.getLogger()

class JointDSRDataset(Dataset):

    def __init__(self,
                 query_tokenizer: T = None,
                 ctx_tokenizer: T = None,
                 data_path: str= None,
                 max_len: int = 512,
                 num_hard_negs: int = 2,
                 train: bool = True,
                 train_mode: str = 'JOINT',
                 use_weighted_ce_loss: bool = False,
                 weighted_sampling: bool = False,
                 target_nli_distribution: List[int] = [1, 1, 1],
                 seed: int = 3,
                ):
        
        super().__init__()

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.NLI_LABEL_MAP = {'SUPPORTS': 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        
        self.query_tokenizer = query_tokenizer if query_tokenizer else ctx_tokenizer
        self.ctx_tokenizer = ctx_tokenizer if ctx_tokenizer else query_tokenizer
        self.max_len = max_len
        self.num_hard_negs = num_hard_negs
        self.train = train
        
        self.train_mode = train_mode

        print(f"Loading data from {data_path}")
        if data_path.endswith(".pkl"):
            self.data = load_pickle(data_path)
        else:
            self.data = [json.loads(line) for line in open(data_path).readlines()]
        print(f"Total sample count: {len(self.data)}")

        self.sampler = None
        self.nli_class_weights = None

        self.target_nli_distribution = target_nli_distribution

        if train_mode == 'JOINT' and self.train:
            self.preprocess()
            if use_weighted_ce_loss:
                self.nli_class_weights = self.get_nli_class_weights()

            if weighted_sampling:
                self.sampler = self.get_weighted_data_sampler()

        print(f"self.sampler: {self.sampler}")
        print(f"self.nli_class_weights: {self.nli_class_weights}")
        
    def preprocess(self):
        processed_data = []
        for d in tqdm(self.data, desc='Preprocessing data by removing examples without enough hard negtive ctx...'):
            neg_num = len(d['hard_negative_ctxs'])
            if neg_num < self.num_hard_negs:
                continue
            else:
                processed_data.append(d)
        self.data = processed_data
        print(f"Total sample count after preprocessing: {len(self.data)}")

    def get_class_cnt(self):
        sup_cnt = 0
        ref_cnt = 0
        nei_cnt = 0

        for d in self.data:
            if d['answers'][0] == 'SUPPORTS':
                sup_cnt += 1
            if d['answers'][0] == 'REFUTES':
                ref_cnt += 1
            if d['answers'][0] == 'NOT ENOUGH INFO':
                nei_cnt += 1

        print(f"sup_cnt: {sup_cnt}")
        print(f"ref_cnt: {ref_cnt}")
        print(f"nei_cnt: {nei_cnt}")

        assert sup_cnt + ref_cnt + nei_cnt == len(self.data)
        return sup_cnt, ref_cnt, nei_cnt

    def get_nli_class_weights(self):
        ep = 1e-5
        sup_cnt, ref_cnt, nei_cnt = self.get_class_cnt()

        logger.info(f"self.target_nli_distribution: {self.target_nli_distribution}")
        class_weights = [self.target_nli_distribution[0]/(sup_cnt+ep), self.target_nli_distribution[1]/(ref_cnt+ep), self.target_nli_distribution[2]/(nei_cnt+ep)]
        return class_weights

    def get_weighted_data_sampler(self):
        if self.nli_class_weights != None:
            class_weights = self.nli_class_weights
        else:
            class_weights = self.get_nli_class_weights()
        
        sample_weights = [0] * len(self.data)
        for idx, d in tqdm(enumerate(self.data), desc='Getting weighted sampler based on distribution of NLI classes...'):
            label = d['answers'][0]
            class_weight = class_weights[self.NLI_LABEL_MAP[label]]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, 
                                        num_samples=len(sample_weights), 
                                        replacement=True) 
        # If we set replacement=False, we will only see that example once.
        # But, since we are doing oversampling, we want it to be True.
        return sampler
                
    def encode_ctx(self, ctx):
        if ctx["title"] != "":
            new_ctx = " . ".join([ctx["title"].strip(), ctx["text"].strip()]) # the sentences in the corpus are encoded like this.
        else:

            new_ctx = ctx["text"].strip()

        return self.ctx_tokenizer.encode(new_ctx.strip(),
                                            max_length=self.max_len,
                                            truncation=True,
                                            padding='max_length', 
                                            return_tensors="pt")
    
    def __getitem__(self, index):
        
        pos_ctx_codes_list = [] # [{'input_ids': [], 'attention_masks': [], }]
        neg_ctx_codes_list = []
        
        sample = self.data[index]
        sample["hard_negative_ctxs"] = sample["hard_negative_ctxs"][:min(2*self.num_hard_negs, len(sample["hard_negative_ctxs"]))]
        query = sample['question']

        if query.endswith("?"):
            query = query[:-1]
        
        q_codes = self.query_tokenizer.encode(query, 
                                              max_length=self.max_len,
                                              truncation=True,
                                              padding='max_length', 
                                              return_tensors="pt")
        
        output = {"q_codes": {},
                  "pos_ctx_codes_list": [],
                  "neg_ctx_codes_list": [],
                  "nli_label": ""}
        
        if self.train:
            random.shuffle(sample["positive_ctxs"])
            random.shuffle(sample["hard_negative_ctxs"])
        
        for pos_ctx in sample['positive_ctxs']:
            pos_ctx_codes = self.encode_ctx(pos_ctx)
            pos_ctx_codes_list.append(pos_ctx_codes)

        if len(pos_ctx_codes_list) == 0:
            logger.info(f"WARNING: len(pos_ctx_codes_list) == 0.")
            logger.info(f"query: {query}")
            logger.info(f"sample['positive_ctxs']: {sample['positive_ctxs']}")
        
        for neg_ctx in sample['hard_negative_ctxs'][:self.num_hard_negs]:
            neg_ctx_codes = self.encode_ctx(neg_ctx)
            neg_ctx_codes_list.append(neg_ctx_codes)

        if len(neg_ctx_codes_list) == 0:
            logger.info(f"WARNING: len(neg_ctx_codes_list) == 0.")
            logger.info(f"query: {query}")
            logger.info(f"sample['hard_negative_ctxs']: {sample['hard_negative_ctxs']}")

        output['q_codes'] = q_codes
        output['pos_ctx_codes_list'] = pos_ctx_codes_list
        output['neg_ctx_codes_list'] = neg_ctx_codes_list

        if self.train_mode == 'JOINT' or self.train_mode == 'NLI_ONLY':
            output['nli_label'] = self.NLI_LABEL_MAP[sample['answers'][0]]

        return output

    def __len__(self):
        return len(self.data)


BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "nli_labels",
        "is_positive",
        "hard_negatives",
        "is_positive_mask",
    ],
)

def batch_collate(samples, pad_id=0, train_mode: str='JOINT', num_hard_negs: int = 2):
    if len(samples) == 0:
        return {}
        
    question_tensors = []
    ctx_tensors = []
    
    positive_ctx_indices = []
    hard_neg_ctx_indices = []
    
    question_segments = []
    ctx_segments = []
    
    nli_labels = []

    for sample in samples:
        # Positive_ctx_code is already shuffled in the Dataset Class.
        question = sample['q_codes']
        positive_ctx = sample['pos_ctx_codes_list'][0]
        hard_neg_ctxs = sample['neg_ctx_codes_list']
        
        # To make sure that positives are evenly distributed in diferent devices when training with multiple gpus.
        # For example, when examples in the first half of the batch have fewer hard-neg ctx, the first half ctx vector will have more pos ctx.
        # As a result, when distributing data into two gpus, the fist gpu will assigned the firt half of the ctx vector in which has more pos ctx.
        # An error will rise when query_num != pos_ctx_num when doing joint training, because we need to do NLI by concatenating query embeddings and the pos ctx embeddings.
        # Note that the batch size may be smaller than the original but it doesn't matter, only has trivial effect to the training speed.
        if train_mode == 'JOINT':
            if len(hard_neg_ctxs) < num_hard_negs:
                continue 
        
        question_tensors.append(question)

        all_ctxs = [positive_ctx] + hard_neg_ctxs

        hard_negatives_start_idx = 1
        hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

        current_ctxs_len = len(ctx_tensors)

        sample_ctxs_tensors = [ctx for ctx in all_ctxs]

        ctx_tensors.extend(sample_ctxs_tensors)
    
        positive_ctx_indices.append(current_ctxs_len)
        hard_neg_ctx_indices.append(
            [
                i
                for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx,
                )
            ]
        )
        
        if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':
            nli_labels.append(sample['nli_label'])

    ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
    questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

    ctx_segments = torch.zeros_like(ctxs_tensor)
    question_segments = torch.zeros_like(questions_tensor)
    
    if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':   
        nli_labels = torch.LongTensor(nli_labels)

    positive_ctx_mask = torch.tensor([0 if i not in positive_ctx_indices else 1 for i in range(ctxs_tensor.shape[0])]).view(ctxs_tensor.shape[0], 1)

    return BiEncoderBatch(
                            questions_tensor,
                            question_segments,
                            ctxs_tensor,
                            ctx_segments,
                            nli_labels,
                            positive_ctx_indices,
                            hard_neg_ctx_indices,
                            positive_ctx_mask
                        )
