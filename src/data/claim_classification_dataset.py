import random

import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from typing import List
import logging

logger = logging.getLogger()

def collate_fn(batch, tokenizer, max_length):
    batch_x, batch_y = [list(x) for x in zip(*batch)]
    batch_x = tokenizer(
                        batch_x,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    )
    batch_y = torch.tensor(batch_y)
    return batch_x, batch_y

class ClaimClassificationDataset(data.Dataset):
    def __init__(self,
                data: List[dict],
                target_nli_distribution: List[int]=[1.0, 1.0, 1.0],
                weighted_sampling: bool=False,
                shuffle_evidence_p: float=0.3,
                train: bool=False,
                ):
        """
        params:
            data: input data, training data contructed from the retriever/reranker's output.
            weighted_sampling: whether to do weighted sampling over the claim classification training examples.
            target_nli_distribution: sampling weights over each class of claim classification examples.
        """
        self.data = data
        self.target_nli_distribution = target_nli_distribution
        self.shuffle_evidence_p = shuffle_evidence_p
        self.train = train
        self.weighted_sampling = weighted_sampling
        
        self.nli_class_weights = self.get_nli_class_weights()
        
        logger.info(f"self.weighted_sampling: {self.weighted_sampling}")
        logger.info(f"self.target_nli_distribution: {self.target_nli_distribution}")
        logger.info(f"self.shuffle_evidence_p: {self.shuffle_evidence_p}")
        logger.info(f"self.train: {self.train}")
        logger.info(f"self.nli_class_weights: {self.nli_class_weights}")

        if weighted_sampling:
            self.sampler = self.get_weighted_data_sampler()
        else:
            self.sampler = None

        logger.info(f"self.sampler: {self.sampler}")

    def get_class_cnt(self):
        sup_cnt = 0
        ref_cnt = 0
        nei_cnt = 0

        for d in self.data:
            if int(d['label']) == 2:
                sup_cnt += 1
            if int(d['label']) == 0:
                ref_cnt += 1
            if int(d['label']) == 1:
                nei_cnt += 1

        logger.info(f"sup_cnt: {sup_cnt}")
        logger.info(f"ref_cnt: {ref_cnt}")
        logger.info(f"nei_cnt: {nei_cnt}")

        assert sup_cnt + ref_cnt + nei_cnt == len(self.data)
        return sup_cnt, ref_cnt, nei_cnt

    def get_nli_class_weights(self):
        ep = 1e-5
        sup_cnt, ref_cnt, nei_cnt = self.get_class_cnt()

        logger.info(f"self.target_nli_distribution: {self.target_nli_distribution}")
        class_weights = [self.target_nli_distribution[0]/(ref_cnt+ep), self.target_nli_distribution[1]/(nei_cnt+ep), self.target_nli_distribution[2]/(sup_cnt+ep)]
        return class_weights

    def get_weighted_data_sampler(self):
        class_weights = self.nli_class_weights
        
        sample_weights = [0] * len(self.data)
        for idx in tqdm(range(len(self.data)), desc='Getting weighted sampler based on distribution of classes...'):
            class_id = self.data[idx]['label']
            class_weight = class_weights[class_id]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, 
                                        num_samples=len(sample_weights), 
                                        replacement=True) 
        # If we set replacement=False, we will only see that example once.
        # But, since we are doing oversampling, we want it to be True.
        return sampler


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):  # concat appoach
        claim = self.data[idx]['claim']
        pred_evidence = self.data[idx]['pred_evi']
        label = int(self.data[idx]['label'])

        logger.debug(f"claim: {claim}")
        logger.debug(f"pred_evidence: {pred_evidence}")
        logger.debug(f"label: {label}")

        pred_evidence = (
            pred_evidence.copy()
        )  # don't want to shuffle original copy, otherwise next epoch the evidence will be out of order

        if (
            self.shuffle_evidence_p
            and self.train
            and random.random() <= self.shuffle_evidence_p
        ):
            random.shuffle(pred_evidence)
            logger.debug(f"pred_evidence: {pred_evidence}")


        formatted_evidence = " </s></s> ".join(
            [
                f"{page_name} -- {evidence_line}"
                for page_name, evidence_line in pred_evidence
            ]
        )
        return (formatted_evidence, claim), label
