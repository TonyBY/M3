# Generate claim classification scores.
import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import logging
from tqdm import tqdm
import numpy as np

import torch

import transformers
transformers.logging.set_verbosity_error()

from src.models.roberta_model import RoBERTa
from src.utils.data_utils import make_directory, load_pickle, process_evid, get_sentence_by_id, get_file_name
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

sep='|#SEP#|'

def get_output_size(args):
    """
    Return the output size of the prediction matix passed on prediction type.
    """
    if args.prediction_type == "singleton":
        return 5
    if args.prediction_type == "mixed":
        return 6
    return 1

def get_inputs(single_sentences, concated_sentence, args):
    """
    Return inputs based on prediction type.
    """
    if args.prediction_type == "singleton":
        return single_sentences
    if args.prediction_type == "mixed":
        return single_sentences + concated_sentence
    return concated_sentence

def write_claim_scores(model, args):
    logger.info(f"Loading dataset from: {args.final_retrieval_results_path}")
    dataset = load_pickle(args.final_retrieval_results_path)
    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    output_size = get_output_size(args)

    all_top5_inputs = []
    all_top5_retrieval_scores = []
    for item in tqdm(dataset, desc="Preparing data..."):
        retrieved_evidence = item[args.retrieved_evidence_feild][:5]
        ids = [
                evi['id'] if 'id' in evi else sep.join([evi[0], str(evi[1])]) for evi in retrieved_evidence
            ]
        
        top5_retrieval_scores =  np.array([float(evi['score']) if 'score' in evi else float(evi[2]) for evi in retrieved_evidence])
        mean_top5_retrieval_scores = [top5_retrieval_scores.mean()]

        if args.prediction_type == "mixed":
            top5_retrieval_scores = (
                top5_retrieval_scores.tolist() + mean_top5_retrieval_scores
            )
        elif args.prediction_type == "concat":
            top5_retrieval_scores = mean_top5_retrieval_scores
        else:
            top5_retrieval_scores = top5_retrieval_scores.tolist()
        all_top5_retrieval_scores.append(top5_retrieval_scores)

        evidence = []
        for sentId in ids:
            evi_text = get_sentence_by_id(sentId, wiki_line_dict)

            if evi_text.split('\t')[0].isdigit():
                evidence.append([process_evid(sentId.split(sep)[0]), process_evid(evi_text.split('\t')[1])])
            else:
                evidence.append([process_evid(sentId.split(sep)[0]), process_evid(evi_text)])

        concat_inputs = " </s></s> ".join(
            [
                page_name + " -- " + sentence
                for (page_name, sentence) in evidence
            ]
        )
        single_sentences = [
            (page_name + " -- " + sentence, item['claim'])
            for page_name, sentence in evidence
        ]

        inputs = get_inputs(single_sentences, [(concat_inputs, item['claim'])], args)

        tokenized_inputs = model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        all_top5_inputs.append(tokenized_inputs)

    preds = []
    softmax_scores = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        with torch.no_grad():
            for input_ids in tqdm(all_top5_inputs, desc="Making predicitons..."):
                input_ids["input_ids"] = input_ids["input_ids"].cuda()
                input_ids["attention_mask"] = input_ids["attention_mask"].cuda()
                pred = model(input_ids)
                softmax_preds = softmax(pred.logits)
                label_pred = (
                    torch.argmax(torch.mean(softmax_preds, dim=0), dim=0)
                    .detach()
                    .cpu()
                    .item()
                )
                softmax_score = softmax_preds.detach().cpu().numpy()
                nan_padded = np.ones((output_size, 3))
                nan_padded[nan_padded == 1] = np.nan
                if softmax_score.shape[0] >= 1:
                    nan_padded[-softmax_score.shape[0] :] = softmax_score
                softmax_scores.append(nan_padded)
                preds.append(label_pred)
                del input_ids

    data = np.concatenate(
        (
            np.array(softmax_scores),
            np.array(all_top5_retrieval_scores).reshape(-1, output_size, 1),
        ),
        axis=2,
    )

    np.save(os.path.join(args.claim_classification_dir, 'cc_' + get_file_name(args.final_retrieval_results_path)  + '.npy'), data)

def main(args):
    logger.info(f"args.model_type: {args.model_type}")
    model = RoBERTa(
            model_type=args.model_type,
            num_labels=3,
        )
    logger.info(f"initiating model parameters from:: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=torch.device("cpu"))["state_dict"]
    if "roberta.loss_fct.weight" in ckpt:
        del ckpt["roberta.loss_fct.weight"]
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()
    write_claim_scores(model, args)

if __name__ == "__main__":
    make_directory(args.claim_classification_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.claim_classification_dir, f"cc_eval.log"))
    logger.info(args)
    main(args)
    logger.info("All Done.")
    