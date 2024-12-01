from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from typing import Tuple, List

import torch.nn.functional as F
from torch import Tensor as T

import torch
import torch.nn as nn

from src.data.joint_dsr_dataset import BiEncoderBatch
import logging

logger = logging.getLogger()

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp, sim_type: str='dot'):
        super().__init__()
        self.temp = temp
        self.sim_type = sim_type
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        logger.info("##########################################")
        logger.info(f"x.shape: {x.shape}")
        logger.info(f"y.shape: {y.shape}")

        logger.info(f"self.temp: {self.temp}")

        if self.sim_type == 'dot':
            logger.info(f"torch.transpose(y, 0, 1).shape: {torch.transpose(y, 0, 1).shape}")
            return torch.matmul(x, torch.transpose(y, 0, 1)) / self.temp
        elif self.sim_type == 'cosine':
            logger.info(f"x.unsqueeze(1).shape: {x.unsqueeze(1).shape}")
            logger.info(f"y.unsqueeze(0).shape: {y.unsqueeze(0).shape}")
            return self.cos(x.unsqueeze(1), y.unsqueeze(0)) / self.temp
        else:
            raise Exception(f"Unsupported similarity type: {self.sim_type}")

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        positive_weight: float = None,
        sim_type: str='dot', 
        temp: float=1.0,
        CROSSENTROPY_LOSS: bool=False,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors, sim_type=sim_type, temp=temp)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        if CROSSENTROPY_LOSS:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                            scores, 
                            torch.tensor(positive_idx_per_question).to(scores.device),
                            )
        else:
            if positive_weight:
                positive_num = softmax_scores.size()[0]
                negative_number = softmax_scores.size()[1] - 1

                positive_weight = 1/(2 * positive_num)
                negative_weight = 1/(2 * negative_number)

                weight_list = [positive_weight if i in positive_idx_per_question else negative_weight for i in range(softmax_scores.size()[1])]
                                

                weight_tensor = torch.FloatTensor(weight_list).to(softmax_scores.device)
            else:
                weight_tensor = None

            loss = F.nll_loss(
                softmax_scores,
                torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                reduction="mean",
                weight=weight_tensor
            )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T, sim_type: str='dot', temp: float=1.0) -> T:
        sim = Similarity(temp, sim_type=sim_type)
        return sim(q_vector, ctx_vectors)

    @staticmethod
    def swap_values_of_different_position_in_tensor(tensor, idx1, idx2):
        temp = tensor[idx1].detach().clone()
        tensor[idx1] = tensor[idx2]
        tensor[idx2] = temp

    @staticmethod
    def move_pos_ctx_to_leftmost(softmax_scores, positive_idx_per_question):
        bz = len(positive_idx_per_question)
        for i in range(bz):
            if i == 0:
                continue
            BiEncoderNllLoss.swap_values_of_different_position_in_tensor(softmax_scores, 
                                                                        (i, 0), 
                                                                        (i, positive_idx_per_question[i]))


def get_contrastive_loss(q_vector: T, 
                         ctx_vectors: T, 
                         batch: BiEncoderBatch, 
                         positive_weight: float=None, 
                         sim_type: str='dot', 
                         temp: float=1.0,
                         CROSSENTROPY_LOSS: bool=False,
                         ) -> Tuple[T, T]:
    
    loss_function = BiEncoderNllLoss()

    loss, is_correct = loss_function.calc(
                                            q_vector,
                                            ctx_vectors,
                                            batch.is_positive,
                                            batch.hard_negatives,
                                            positive_weight=positive_weight,
                                            sim_type=sim_type, 
                                            temp=temp,
                                            CROSSENTROPY_LOSS=CROSSENTROPY_LOSS,
                                        )
    return loss, is_correct

def get_nli_loss(nli_logits: T, nli_labels: T, nli_class_weights: List[int] = None) -> Tuple[T, T, T, T]:
    if nli_class_weights:
        nli_weight_tensor = torch.FloatTensor(nli_class_weights).to(nli_logits.device)
    else:
        nli_weight_tensor=None

    logger.info(f"nli_weight_tensor: {nli_weight_tensor}")

    loss_fct = nn.CrossEntropyLoss(weight=nli_weight_tensor)

    activation_fct = nn.Softmax(dim=1)
    softmax_scores = activation_fct(nli_logits)
    max_score, max_idxs = torch.max(softmax_scores, 1)
    correct_predictions_count = (max_idxs == nli_labels.clone().to(max_idxs.device)).sum()
    loss = loss_fct(softmax_scores, nli_labels)

    return softmax_scores, nli_labels, loss, correct_predictions_count

def multi_task_loss(batch: BiEncoderBatch, 
                    model_output: Tuple[T, T, T, T], 
                    train_mode:str = 'JOINT', 
                    positive_weight: float=None,
                    nli_class_weights: List[int] = None,
                    retrieval_to_nli_weight: float = 1,
                    sim_type: str='dot', 
                    temp: float=1.0,
                    CROSSENTROPY_LOSS: bool=False,
                    ):
    query_embeddings, ctx_embeddings, nli_logits, nli_labels = model_output
    
    if train_mode == 'JOINT' or train_mode == 'RETRIEVE_ONLY':
        con_loss, con_correct_num = get_contrastive_loss(query_embeddings, 
                                                         ctx_embeddings, 
                                                         batch, 
                                                         positive_weight=positive_weight, 
                                                         sim_type=sim_type, 
                                                         temp=temp,
                                                         CROSSENTROPY_LOSS=CROSSENTROPY_LOSS,
                                                         )
    else:
        con_loss = 0
        con_correct_num = 0
    
    logger.info(f"train_mode: {train_mode}")
    logger.info(f"nli_class_weights: {nli_class_weights}")
    if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':
        nli_softmax_scores, nli_labels, nli_loss, nli_correct_num = get_nli_loss(nli_logits, nli_labels, nli_class_weights=nli_class_weights)
    else:
        nli_loss = 0
        nli_correct_num = 0

    if retrieval_to_nli_weight >= 1:
        mt_loss = con_loss + (1 / retrieval_to_nli_weight) * nli_loss
    elif retrieval_to_nli_weight > 0:
        mt_loss = retrieval_to_nli_weight * con_loss + nli_loss
    else:
        logger.info(f"Illegal value for retrieval_to_nli_weight: {retrieval_to_nli_weight}. It has to be positive. Setting retrieval_to_nli_weight to 1.")
        mt_loss = con_loss + nli_loss
    return con_loss, con_correct_num, nli_loss, nli_correct_num, mt_loss

def batch_eval(batch: BiEncoderBatch, output: Tuple[T, T, T, T]) -> List[float]:
    query_embeddings, ctx_embeddings, nli_logits, nli_labels = output

    rrs = []
    if query_embeddings != None:
        scores = BiEncoderNllLoss.get_scores(query_embeddings, ctx_embeddings)
        logger.debug(f"scores.shape: {scores.shape}")
        logger.debug(f"scores: {scores}")

        if len(query_embeddings.size()) > 1:
            q_num = query_embeddings.size(0)
            scores = scores.view(q_num, -1)
            logger.debug(f"scores.view(q_num, -1).shape: {scores.shape}")
            logger.debug(f"scores.view(q_num, -1): {scores}")

        ranked = scores.argsort(dim=1, descending=True)
        logger.debug(f"ranked.shape: {ranked.shape}")
        logger.debug(f"ranked: {ranked}")
        idx2ranked = ranked.argsort(dim=1)

        target = batch.is_positive
        logger.debug(f"idx2ranked.shape: {idx2ranked.shape}")
        logger.debug(f"idx2ranked: {idx2ranked}")
        logger.debug(f"target: {target}")

        for t, i2r in zip(target, idx2ranked):
            rrs.append(1 / (i2r[t].item() + 1))
            logger.debug(f"t: {t}")
            logger.debug(f"i2r.shape: {i2r.shape}")
            logger.debug(f"i2r: {i2r}")
            logger.debug(f"i2r[t].item(): {i2r[t].item()}")

    _, nli_preds = torch.max(nli_logits, 1) if nli_logits != None else (0, [])
    
    if nli_logits != None:
        return {"rrs": rrs, "nli_preds": nli_preds.to('cpu'), "nli_labels": nli_labels.to('cpu')}
    else:
        return {"rrs": rrs, "nli_preds": nli_preds, "nli_labels": nli_labels}
