import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import logging
from tqdm import tqdm
import time
from typing import List

import torch
import torch.nn.functional as F
from torch import autocast
from torch import Tensor as T

import transformers
from transformers import AutoTokenizer
transformers.logging.set_verbosity_error()

from src.models.roberta_model import RoBERTa
from src.utils.data_utils import make_directory, load_pickle, dump_pickle, \
                                 process_evid, get_sentence_by_id, get_file_name
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

sep='|#SEP#|'

def get_docids_from_a_list_of_preds(preds:List[dict]=None, sep: str='|#SEP#|') -> List[str]:
    pred_doc_ids = []
    viewed_doc_ids = set()
    for pred in preds:
        logger.debug(f"pred: {pred}")
        doc_id = pred['id'].split(sep)[0]
        if doc_id not in viewed_doc_ids:
            viewed_doc_ids.add(doc_id)
            pred_doc_ids.append(doc_id)
        else:
            continue
    assert len(pred_doc_ids) == len(viewed_doc_ids)
    return pred_doc_ids

def load_model(model_type: str=None, model_path: str=None, num_labels: int=None,):
    logger.info(f"Preparing model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = RoBERTa(model_type, num_labels, tokenizer=tokenizer)
    model.expand_embeddings()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    model.eval()
    logger.info("Done.")
    logger.info("{}: {:.2f}s".format(0, (time.time() - t0)))
    return model, tokenizer

def get_topk_reranked_sentences(model=None,
                                tokenizer: AutoTokenizer=None,
                                claim: str=None, 
                                lines: List[List[str]] = None, 
                                rerank_topk: int=5, 
                                batch_size: int=None, 
                                ):
    """
    Output: 
        topk_preds: a list of index number of evidence in the candidate sentence set.
        score_criteria: a tensor of scores of the reranked sentences.
    """
    i = 0
    all_preds = []
    with autocast(dtype=torch.bfloat16, device_type="cuda"):
        with torch.no_grad():
            while i < len(lines):
                batch_lines = lines[i : i + batch_size]
                
                # Batch Input: A list of (title -- line, claim) pairs.
                batch_inputs = [
                                ((title + " -- " + line), claim)
                                for title, line in batch_lines
                            ]
                
                # Tokenized batch input
                batch_input_ids = tokenizer(
                                batch_inputs,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=256,
                            )
                
                # Move tokenized batch input to gpu
                batch_input_ids["input_ids"] = batch_input_ids[
                                                                "input_ids"
                                                            ].cuda()

                batch_input_ids["attention_mask"] = batch_input_ids[
                                "attention_mask"
                            ].cuda()
                
                preds = model(batch_input_ids).logits.float().detach().cpu()
                del batch_input_ids
                
                all_preds += [preds]
                i = i + batch_size

    # sofrmaxt scores for each label of each evidence.           
    all_preds = torch.cat(all_preds, dim=0)
    softmax_scores = F.softmax(all_preds, dim=1)
    
    if args.num_labels == 3:
        score_criteria = 1 - softmax_scores[:, 1]
    else:
        score_criteria = softmax_scores[:, 1]
                
    args_sort = torch.argsort(score_criteria, descending=True)
    topk_preds = args_sort[:rerank_topk]
    return topk_preds, score_criteria


def get_lines(evidence_list: List[dict]=None, 
              wiki_line_dict: dict=None,
              ):
    lines = []
    for evi in evidence_list:
        sentId = evi['id']
        evi_text = get_sentence_by_id(sentId, wiki_line_dict)
        if evi_text == '':
            logger.info(f"WARNING: Sentence: {evi['id']} not in wiki_line_dict.")
            continue
        if evi_text == '':
            logger.info(f"WARNING: Sentence: {evi['id']} not in wiki_line_dict.")
            continue
            
        if '\t' in evi_text:
            if evi_text.split('\t')[0].isdigit():
                lines.append([process_evid(evi['id'].split(sep)[0]), process_evid(evi_text.split('\t')[1])])
            else:
                lines.append([process_evid(evi['id'].split(sep)[0]), process_evid(evi_text)])
        else:
            lines.append([process_evid(evi['id'].split(sep)[0]), process_evid(evi_text)])

    return lines

def main(args):
    model, tokenizer = load_model(
                                    model_type=args.model_type,
                                    model_path=args.model_path,
                                    num_labels=args.num_labels,
                                )
    
    logger.info(f"Loading first_hop_search_results from: {args.first_hop_search_results_path}")
    first_hop_search_results = load_pickle(args.first_hop_search_results_path)
    if args.debug:
        first_hop_search_results = first_hop_search_results[:10]
    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    logger.info(f"Start ranking sentences in the top docs for each claim.")
    t0 = time.time()
    multi_hop_reranking_results = []
    for item in tqdm(first_hop_search_results, desc=f"Evaluating sentence reranker: {args.model_type}"):    
        evidence_list = []
        viewed_sentence_ids = set()
        
        if 'sufficiency_checking_results' in item:
            reranked_evi = item['sufficiency_checking_results']
        elif 'reranked_context' in item:
            reranked_evi = item['reranked_context']
        elif 'multihop_context' in item:
                reranked_evi = item['context'][:args.singleHopNumbers]
        else:
            reranked_evi = []

        if reranked_evi != []:
            try:
                reranked_evi_ids = [sep.join([str(evi[0]), str(evi[1])]) for evi in reranked_evi]
                reranked_evi = [{'id': sep.join([str(evi[0]), str(evi[1])]), 'score': float(evi[2])} for evi in reranked_evi]
            except:
                 for i in range(len(reranked_evi)):
                    reranked_evi_ids = [evi['id'] for evi in reranked_evi]
            logger.info(f"len(reranked_evi): {len(reranked_evi)}")
            logger.debug(f"reranked_evi: {reranked_evi}")
            viewed_sentence_ids.update(reranked_evi_ids)
        else:
            raise Exception("Error: reranked_evi == []! Please do single-hop sentence reranking before doing multi-hop sentence reranking.")    

        context = item['multihop_context']

        for c in context:
            if c['id'] not in viewed_sentence_ids:
                viewed_sentence_ids.add(c['id'])
                evidence_list.append(c)
            if len(evidence_list) == args.fist_hop_topk:
                break

        logger.debug(f"len(evidence_list): {len(evidence_list)}")
        logger.debug(f"evidence_list: {evidence_list}")

        lines = get_lines(evidence_list = evidence_list, 
                          wiki_line_dict = wiki_line_dict,
                        )
        if lines == []:
            logger.info(f"WARNING: No evidence is available.")
            continue

        reranked_evi_lines = get_lines(evidence_list = reranked_evi, 
                          wiki_line_dict = wiki_line_dict,
                        )
        
        evi_sentence_strs = [' . '.join(line) for line in reranked_evi_lines]
        if args.concat_claim:
            claims = [" -- ".join([item['claim'], evi]) for evi in evi_sentence_strs]
        else:
            claims = evi_sentence_strs
        logger.debug(f"len(claims) {len(claims)}")
        logger.debug(f"claims: {claims}")

        first_hop_evidence_scores = [evi['score'] for evi in reranked_evi]
        first_hop_evidence_ids = [evi['id'] for evi in reranked_evi]
        
        item['multihop_reranked_context'] = []
        for i in range(len(claims)):
            claim = claims[i]
            first_hop_evidence_score = first_hop_evidence_scores[i]
            first_hop_evidence_id = first_hop_evidence_ids[i]
            topk_preds, score_criteria = get_topk_reranked_sentences(
                                                                    model = model,
                                                                    tokenizer = tokenizer,
                                                                    claim = claim, 
                                                                    lines = lines, 
                                                                    rerank_topk = args.rerank_topk, 
                                                                    batch_size = args.retrank_batch_size, 
                                                                    )

            topk_evidence = []
            for top in topk_preds:
                second_hop_evidence_score = float(score_criteria[top])
                if args.save_evi_path:
                    topk_evidence.append([first_hop_evidence_id, first_hop_evidence_score, evidence_list[top]['id'], second_hop_evidence_score])
                else:
                    topk_evidence.append({'id': evidence_list[top]['id'], 'score': second_hop_evidence_score}) # Note, there might be duplicate evidence ids with different scores.
            item['multihop_reranked_context'].extend(topk_evidence)
        
        multi_hop_reranking_results.append(item)

    logger.info(f"Saving results")
    t0 = time.time()
    data_name = get_file_name(args.first_hop_search_results_path)
    if args.debug:
        data_name = "DEBUG_" + data_name

    dump_pickle(
        multi_hop_reranking_results,
        os.path.join(args.reranking_dir, f"{data_name}_srrmTop-{args.fist_hop_topk}_savedTop{args.rerank_topk}_savePath{args.save_evi_path}.pkl"),
    )

    logger.info("Done.")
    logger.info("{}: {:.2f}s".format(0, (time.time() - t0)))


if __name__ == "__main__":
    make_directory(args.reranking_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.reranking_dir, f"reranker_eval_{args.fist_hop_topk}.log"))
    logger.info(args)
    main(args)
    