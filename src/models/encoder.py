import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from pyserini.encode import DprQueryEncoder
from typing import List, Union
import torch

from transformers import AutoTokenizer

from src.utils.data_utils import move_to_device

class DprQueryBatchEncoder(DprQueryEncoder):
    def __init__(self, max_length: int = 70, **kwargs):
        """
        param: max_length: int, the maximum length in number of tokens for the inputs to the transformer model.
        """
        super().__init__(**kwargs)
        self.max_length = max_length

        
    def encode_batch(self, batch_q: List[str]):
        batch_q_encodes = self.tokenizer.batch_encode_plus(batch_q, max_length=self.max_length, 
                                                              padding='max_length', 
                                                              truncation=True, 
                                                              return_tensors="pt")
        batch_q_encodes = move_to_device(dict(batch_q_encodes), self.model.device)
        input_ids, q_mask, q_type_ids = batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None)
        with torch.no_grad():
            embeddings = self.model(input_ids, q_mask, q_type_ids).pooler_output.detach().cpu().numpy()
        return embeddings


class SentenceEncoder(object):
    
    def __init__(self, model,
                 tokenizer_name_or_path:str=None, 
                 max_length:int=512,
                 sentence_type:str="ctx"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            
        try:
            self.encoder = model.encoder
        except:
            if sentence_type == "ctx":
                self.encoder = model.ctx_encoder
            elif sentence_type == "query":
                self.encoder = model.query_encoder
        
        print(f"type(self.encoder): {type(self.encoder)}")
        self.max_length = max_length
    
    def encode_batch(self, sentence_list: Union[str, List[str], List[dict]], titles=True, sep='|#SEP#|'):
        if len(sentence_list) == 0:
            raise Exception("Length of input sentence list cannot be empty.")
        if isinstance(sentence_list[0], dict):
            try:
                sentence_list = [s['contents'] for s in sentence_list] # Doc title is already included in the context during preprocessing.
            except KeyError:
                sentence_list = [s['context'] for s in sentence_list]
                
        batch_s_encodes = self.tokenizer.batch_encode_plus(sentence_list,
                                                           max_length=self.max_length, 
                                                           padding='max_length', 
                                                           truncation=True, 
                                                           return_tensors="pt")
        batch_s_encodes = move_to_device(dict(batch_s_encodes), self.encoder.device)
        input_ids, s_mask, s_type_ids = batch_s_encodes["input_ids"], batch_s_encodes["attention_mask"], batch_s_encodes.get("token_type_ids", None)
        with torch.no_grad():
            embeddings = self.encoder(input_ids, s_mask, s_type_ids).pooler_output.detach().cpu()
        return embeddings
        