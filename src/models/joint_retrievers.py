from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from typing import Tuple

from transformers import (AutoModel, 
                          DPRQuestionEncoder, 
                          DPRContextEncoder, )

import torch
from torch import Tensor as T
import torch.nn as nn
import logging

logger = logging.getLogger()

class MLPClassificationHead(nn.Module):
        """Head for sentence-level classification tasks."""

        def __init__(self, config):
            super().__init__()
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.m = nn.GELU()
            self.layer1  = nn.Linear(config.hidden_size * 2, 512)
            self.layer2  = nn.Linear(512, 128)
            self.out_proj1 = nn.Linear(128, config.num_labels)

        def forward(self, x, **kwargs):
            """
            x: pooler output, i.e., embedding of <s> token (equiv. to [CLS]) == ecoder(input)[:, 0, :]
            """
            logger.info(f"x.shape: {x.shape}")
            x = self.dropout(x)
            x = self.layer1(x)
            x = self.m(x)
            x = self.layer2(x)
            x = self.m(x)
            x = self.out_proj1(x)
            return x

class BiEncoderRetriever(nn.Module):
    def __init__(self, 
                 query_encoder_dir: str = None, 
                 ctx_encoder_dir: str = None, 
                 n_classes: int = 3,
                 share_encoder = False,
                 ):
    
        super().__init__()
                
        if 'dpr' in ctx_encoder_dir.lower():
            self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_dir)
        else:
            self.ctx_encoder = AutoModel.from_pretrained(ctx_encoder_dir)

        self.ctx_encoder
        
        if share_encoder:
            self.query_encoder = self.ctx_encoder
        else:
            if 'dpr' in query_encoder_dir.lower():
                self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_dir)
            else:
                self.query_encoder = AutoModel.from_pretrained(query_encoder_dir)

            self.query_encoder
                
        config = self.ctx_encoder.config
        config.num_labels = n_classes

        self.nli_head = MLPClassificationHead(config)
    
    def forward(
        self,
        query_input_ids: T,
        query_attention_mask: T,
        query_token_type_ids: T,
        ctx_input_ids: T,
        ctx_attention_mask: T,
        ctx_token_type_ids: T,
        positive_ctx_mask: T, # positive ctx indices in binary, e.g., [0, 3] -> [1, 0, 0, 1, 0, 0]
        nli_labels: T,
        encoder_type: str = None,
        train_mode='RETRIEVE_ONLY',
    ) -> Tuple[T, T, T]:
        query_encoder = self.query_encoder if encoder_type is None or encoder_type == "query" else self.ctx_encoder
        ctx_encoder = self.ctx_encoder if encoder_type is None or encoder_type == "ctx" else self.query_encoder
        
        query_embeddings = query_encoder(query_input_ids,
                                        query_attention_mask,
                                        query_token_type_ids).pooler_output

        ctx_embeddings = ctx_encoder(ctx_input_ids,
                                    ctx_attention_mask,
                                    ctx_token_type_ids).pooler_output

        if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':
            positive_ctx_indices = positive_ctx_mask.view(-1).nonzero().contiguous().view(-1)
            positive_ctx_embeddings = torch.stack([ctx_embeddings[i] for i in positive_ctx_indices], dim=0)
            nli_embeddings = torch.cat((query_embeddings, positive_ctx_embeddings), -1)
            logger.info(f"nli_embeddings.shape: {nli_embeddings.shape}")
            nli_logits = self.nli_head(nli_embeddings)
        else:
            nli_logits = None
        return query_embeddings, ctx_embeddings, nli_logits, nli_labels

class SingleEncoderRetriever(nn.Module):
    def __init__(self, 
                 encoder_dir: str = None, 
                 n_classes: int = 3,
                 ):
    
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_dir)
        self.encoder

        config = self.encoder.config
        config.num_labels = n_classes

        self.nli_head = MLPClassificationHead(config)
    
    def forward(
        self,
        query_input_ids: T,
        query_attention_mask: T,
        query_token_type_ids: T,
        ctx_input_ids: T,
        ctx_attention_mask: T,
        ctx_token_type_ids: T,
        positive_ctx_mask: T,
        nli_labels: T,
        encoder_type: str = None,
        train_mode='RETRIEVE_ONLY',
    ) -> Tuple[T, T, T]:
        query_embeddings = self.encoder(query_input_ids,
                                        query_attention_mask,
                                        query_token_type_ids).pooler_output
            
        ctx_embeddings = self.encoder(ctx_input_ids,
                                    ctx_attention_mask,
                                    ctx_token_type_ids).pooler_output
        
        if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':
            positive_ctx_indices = positive_ctx_mask.view(-1).nonzero().contiguous().view(-1)
            positive_ctx_embeddings = torch.stack([ctx_embeddings[i] for i in positive_ctx_indices], dim=0)

            nli_embeddings = torch.cat((query_embeddings, positive_ctx_embeddings), -1)
            logger.info(f"nli_embeddings.shape: {nli_embeddings.shape}")
            nli_logits = self.nli_head(nli_embeddings)
        else:
            nli_logits = None
        
        return query_embeddings, ctx_embeddings, nli_logits, nli_labels
        