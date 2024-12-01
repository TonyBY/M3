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

import transformers
from transformers import AutoTokenizer
transformers.logging.set_verbosity_error()

from src.models.roberta_model import RoBERTa
from src.utils.data_utils import make_directory, load_pickle, dump_pickle, \
                                 process_evid, get_sentence_by_id, \
                                 get_file_name, pred_format_pyserini_to_fever, get_multiHop_and_singleHop_data
from src.utils.args import prepare_logger
from src.utils.config import parser
from src.eval.ir_evaluator import eval_doc_recall_in_fever_format, custom_eval, evidence_retrieval_score

args = parser.parse_args()
logger = logging.getLogger()

sep='|#SEP#|'

def load_model(model_type: str=None, model_path: str=None, num_labels: int=None,):
    logger.info(f"Preparing model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = RoBERTa(model_type, num_labels, tokenizer=tokenizer)
    try:
        model.expand_embeddings()
    except:
        pass
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
                logger.debug(f"preds.size(): {preds.size()}")
                logger.debug(f"preds: {preds}")
                del batch_input_ids
                
                all_preds += [preds]
                i = i + batch_size

    # sofrmaxt scores for each label of each evidence.           
    all_preds = torch.cat(all_preds, dim=0)
    logger.debug(f"all_preds.size(): {all_preds.size()}")
    logger.debug(f"all_preds: {all_preds}")
    softmax_scores = F.softmax(all_preds, dim=1)
    logger.debug(f"softmax_scores.size(): {softmax_scores.size()}")
    logger.debug(f"softmax_scores: {softmax_scores}")
    
    if args.num_labels == 3:
        score_criteria = 1 - softmax_scores[:, 1]
    else:
        score_criteria = softmax_scores[:, 1]
                
    args_sort = torch.argsort(score_criteria, descending=True)
    # topk_preds = args_sort[:rerank_topk]
    topk_preds = args_sort
    return topk_preds, score_criteria

def main(args):
    model, tokenizer = load_model(
                                    model_type=args.model_type,
                                    model_path=args.model_path,
                                    num_labels=args.num_labels,
                                )
    
    logger.info(f"Loading first_hop_search_results from: {args.first_hop_search_results_path}")
    first_hop_search_results = load_pickle(args.first_hop_search_results_path)
    if args.debug:
        first_hop_search_results = first_hop_search_results[:100]

    logger.info(f"len(first_hop_search_results[0]['context']): {len(first_hop_search_results[0]['context'])}")
    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    print(f"Start ranking sentences in the top docs for each claim.")
    t0 = time.time()
    first_hop_reranking_results = []
    for item in tqdm(first_hop_search_results, desc=f"Evaluating sentence reranker: {args.model_type}"):    
        claim = item['claim']
        if args.joint_reranking:
            evidence_list = item['context'][:args.singleHopNumbers] + item['multihop_context'][:args.multiHopNumbers]
        else:
            evidence_list = item['context'][:args.fist_hop_topk]
        logger.info(f"len(evidence_list): {len(evidence_list)}")
        logger.debug(f"evidence_list: {evidence_list}")
        
        lines = []
        for evi in evidence_list:
            sentId = evi['id']
            evi_text = get_sentence_by_id(sentId, wiki_line_dict)
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

        if lines == []:
            logger.info(f"WARNING: No evidence is available.")
            continue
        
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
            doc_id, line_num = evidence_list[top]['id'].split(sep)
            topk_evidence.append({'id': str(doc_id) + sep + str(line_num), 'score': float(score_criteria[top])})
        item['context'] = topk_evidence
        logger.info(f"len(item['context']): {len(item['context'])}")
        if 'multihop_context' in item:
            item.pop('multihop_context') 
        first_hop_reranking_results.append(item)

    if 'label' in first_hop_reranking_results[0]:
        fever_result = pred_format_pyserini_to_fever(first_hop_reranking_results)

        max_evidence = [1, 5, 10, 20, 40, 50, 100, 200, 300, 400, 500, 1000, 2048, 4096]
   
        data_loose_single, data_strict_single, data_multi_strict_level0, data_multi_strict_level1, data_multi_strict_level2, data_multi_strict_level3 = get_multiHop_and_singleHop_data(fever_result)
        logger.info(f"len(fever_result): {len(fever_result)}")
        logger.info(f"len(data_loose_single): {len(data_loose_single)}")
        logger.info(f"len(data_strict_single): {len(data_strict_single)}")
        logger.info(f"len(data_multi_strict_level0): {len(data_multi_strict_level0)}")
        logger.info(f"len(data_multi_strict_level1): {len(data_multi_strict_level1)}")
        logger.info(f"len(data_multi_strict_level2): {len(data_multi_strict_level2)}")
        logger.info(f"len(data_multi_strict_level3): {len(data_multi_strict_level3)}")

        logger.info(f"[len(fever_result[i]['predicted_evidence']) for i in range(len(fever_result[:100]))]: {[len(fever_result[i]['predicted_evidence']) for i in range(len(fever_result[:100]))]}")
        
        result_type = ['ALL', 'LooseSingleHop', 'StrictSingleHop', 'StrictMultiHop_Level_0', 'StrictMultiHop_Level_1', 'StrictMultiHop_Level_2', 'StrictMultiHop_Level_3']
        for idx, r in enumerate([fever_result, 
                                data_loose_single, 
                                data_strict_single, 
                                data_multi_strict_level0,
                                data_multi_strict_level1,
                                data_multi_strict_level2,
                                data_multi_strict_level3]):
            logger.info(f"\n\n====================================================")
            logger.info(f"================== {result_type[idx]} =================")
            logger.info(f"======================================================")
            for me in tqdm(max_evidence, desc='Evaluating topk retrieval results.'):
                logger.info(f"\n******* retrieval scores @ {me}**********")
                if all([me > len(fever_result[i]['predicted_evidence']) for i in range(len(fever_result[:100]))]):
                    logger.info("all([me > len(fever_result[i]['predicted_evidence']) for i in range(len(fever_result[:100]))])")
                    break   
                pr, rec, f1 = evidence_retrieval_score(r, max_evidence=me)
                logger.info(f"pr: {pr}")
                logger.info(f"rec: {rec}")
                logger.info(f"f1: {f1}")

                eval_doc_recall_in_fever_format(r, max_evidence=me)

                custom_eval(r, max_evidence=me)

    logger.info(f"Saving results")
    t0 = time.time()
    data_name = get_file_name(args.first_hop_search_results_path)
    if args.debug:
        data_name = "DEBUG_" + data_name

    dump_pickle(
        first_hop_reranking_results,
        os.path.join(args.reranking_dir, f"{data_name}_1hop_rerank_Topk-{args.fist_hop_topk}.pkl"),
    )

    logger.info("Done.")
    logger.info("{}: {:.2f}s".format(0, (time.time() - t0)))


if __name__ == "__main__":
    make_directory(args.reranking_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.reranking_dir, f"reranker_eval_{args.fist_hop_topk}.log"))
    logger.info(args)
    main(args)
    logger.info("All Done.")
    