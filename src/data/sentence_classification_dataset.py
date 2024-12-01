import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from typing import List
from tqdm import tqdm
import logging

logger = logging.getLogger()

def collate_fn(batch, tokenizer, max_length):
    batch_evi, batch_nliLabel, batch_sufLabel = [list(x) for x in zip(*batch)]
    batch_evi = tokenizer(
                            batch_evi,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=max_length,
                        )
    batch_nliLabel = torch.tensor(batch_nliLabel)
    batch_sufLabel = torch.tensor(batch_sufLabel)
    return batch_evi, batch_nliLabel, batch_sufLabel

        
class SentenceDatasetRoBERTa(data.Dataset):
    def __init__(self, 
                 data, 
                 tokenizer, 
                 binary_label=False, 
                 claim_second=False, 
                 target_nli_distribution: List[int]=[1.0, 1.0, 1.0],
                 target_suf_distribution: List[int]=[1.0, 1.0, 1.0],
                 SUF_CHECK: bool=False,
                 weighted_sampling: bool=False,
                 ):
        self.data = data
        self.binary_label = binary_label
        self.target_nli_distribution = target_nli_distribution
        self.target_suf_distribution = target_suf_distribution
        self.SUF_CHECK = SUF_CHECK

        if self.SUF_CHECK:
            self.suf_class_weights = self.get_suf_class_weights()
            logger.info(f"self.suf_class_weights: {self.suf_class_weights}")
            self.nli_class_weights=None
        else:
            self.nli_class_weights = self.get_nli_class_weights()
            logger.info(f"self.nli_class_weights: {self.nli_class_weights}")
            self.suf_class_weights=None

        if weighted_sampling:
            self.sampler = self.get_weighted_data_sampler()
        else:
            self.sampler = None
        
    def get_weighted_data_sampler(self):
        if self.nli_class_weights != None:
            class_weights = self.nli_class_weights
        else:
            class_weights = self.suf_class_weights
        
        sample_weights = [0] * len(self.data)
        for idx in tqdm(range(len(self.data)), desc='Getting weighted sampler based on distribution of classes...'):
            if self.SUF_CHECK:
                class_id = self.data.iloc[idx].sufficiency_label
            else:
                class_id = self.data.iloc[idx].nli_label
            class_weight = class_weights[class_id]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, 
                                        num_samples=len(sample_weights), 
                                        replacement=True) 
        # If we set replacement=False, we will only see that example once.
        # But, since we are doing oversampling, we want it to be True.
        return sampler

    def get_nli_class_weights(self):
        ep = 1e-5
        logger.info(f"self.target_nli_distribution: {self.target_nli_distribution}")
        if self.binary_label:
            pos_cnt, neg_cnt = self.get_class_cnt()
            class_weights = [self.target_nli_distribution[0]/(neg_cnt+ep), self.target_nli_distribution[1]/(pos_cnt+ep)]
        else:
            sup_cnt, ref_cnt, nei_cnt = self.get_class_cnt()
            class_weights = [self.target_nli_distribution[0]/(ref_cnt+ep), self.target_nli_distribution[1]/(nei_cnt+ep), self.target_nli_distribution[2]/(sup_cnt+ep)]
        return class_weights
    
    def get_suf_class_weights(self):
        ep = 1e-5
        suf_cnt, insuf_cnt, neg_cnt = self.get_class_cnt()

        logger.info(f"self.target_suf_distribution: {self.target_suf_distribution}")
        class_weights = [self.target_suf_distribution[0]/(insuf_cnt+ep), self.target_suf_distribution[1]/(neg_cnt+ep), self.target_suf_distribution[2]/(suf_cnt+ep)]
        return class_weights
    
    def get_class_cnt(self):
        if self.SUF_CHECK:
            suf_cnt = len(self.data.iloc[self.data.index[self.data.sufficiency_label==2]])
            insuf_cnt = len(self.data.iloc[self.data.index[self.data.sufficiency_label==0]])
            neg_cnt = len(self.data.iloc[self.data.index[self.data.sufficiency_label==1]])

            logger.info(f"suf_cnt: {suf_cnt}")
            logger.info(f"insuf_cnt: {insuf_cnt}")
            logger.info(f"neg_cnt: {neg_cnt}")

            assert suf_cnt + insuf_cnt + neg_cnt == len(self.data)
            return suf_cnt, insuf_cnt, neg_cnt
        else:
            if self.binary_label:
                pos_cnt = len(self.data.iloc[self.data.index[self.data.nli_label==1]])
                neg_cnt = len(self.data.iloc[self.data.index[self.data.nli_label==0]])
                logger.info(f"pos_cnt: {pos_cnt}")
                logger.info(f"neg_cnt: {neg_cnt}")

                assert pos_cnt + neg_cnt == len(self.data)
                return pos_cnt, neg_cnt
            else:
                sup_cnt = len(self.data.iloc[self.data.index[self.data.nli_label==2]])
                ref_cnt = len(self.data.iloc[self.data.index[self.data.nli_label==0]])
                nei_cnt = len(self.data.iloc[self.data.index[self.data.nli_label==1]])

                logger.info(f"sup_cnt: {sup_cnt}")
                logger.info(f"ref_cnt: {ref_cnt}")
                logger.info(f"nei_cnt: {nei_cnt}")

                assert sup_cnt + ref_cnt + nei_cnt == len(self.data)
                return sup_cnt, ref_cnt, nei_cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim, candidate_title, candidate_sentence, nli_label, sufficiency_label = self.data.iloc[idx]
        claim = str(claim)
        claim = claim[0].upper() + claim[1:]
        candidate_title = str(candidate_title)
        candidate_sentence = str(candidate_sentence)
        if candidate_title == "nan":
            return (candidate_sentence, claim), nli_label, sufficiency_label
        else:
            return (candidate_title + " -- " + candidate_sentence, claim), nli_label, sufficiency_label
