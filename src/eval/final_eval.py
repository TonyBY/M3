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

import transformers
transformers.logging.set_verbosity_error()

from src.eval.xgb_classifier import reorder_by_score_filter_add_softmax
from src.eval.ir_evaluator import eval_doc_recall_in_fever_format, custom_eval
from src.eval.fever.scorer import fever_score
from src.utils.data_utils import make_directory, read_jsonl, \
                                 save_jsonl, load_pickle, \
                                 get_file_name, get_multiHop_and_singleHop_data
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()
sep='|#SEP#|'

def predict(args):
    data = load_pickle(args.final_retrieval_results_path)

    top5_predictions = np.load(
        args.claim_scores_path
    )
    
    top5_predictions, _ = reorder_by_score_filter_add_softmax(
        top5_predictions, data
    )

    rfc = load_pickle(args.xgbc_model_path)
    preds = rfc.predict(top5_predictions.reshape((len(top5_predictions), -1)))
    int2sym = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}
    instances = []

    for (item, pred) in tqdm(zip(data, preds)):
        sym_pred = int2sym[pred]
        predicted_evidence = []
        for top_sentence in item[args.retrieved_evidence_feild]:
            predicted_evidence.append([top_sentence['id'].split(sep)[0], int(top_sentence['id'].split(sep)[1])])
        if 'evidence' in item:
            instance = {
                "id": item['id'],
                "claim": item['claim'],
                "label": item['label'],
                "predicted_label": sym_pred,
                "evidence": item['evidence'],
                "predicted_evidence": predicted_evidence,
            }
        else:
            instance = {
                "id": item['id'],
                "claim": item['claim'],
                "predicted_label": sym_pred,
                "predicted_evidence": predicted_evidence,
            }
        logger.debug(f"instance: {instance}")
        instances.append(instance)
    return instances

if __name__ == "__main__":
    make_directory(args.submission_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.submission_dir, f"submission.log"))
    logger.info(args)

    if 'train' in get_file_name(args.claim_scores_path).lower():
        out_put_path = os.path.join(args.submission_dir, "predictions_" + "train" + ".jsonl")
    elif 'dev' in get_file_name(args.claim_scores_path).lower():
        out_put_path = os.path.join(args.submission_dir, "predictions_" + "dev" + ".jsonl")
    elif 'test' in get_file_name(args.claim_scores_path).lower():
        out_put_path = os.path.join(args.submission_dir, "predictions_" + "test" + ".jsonl")
    else:
        out_put_path = os.path.join(args.submission_dir, "predictions_" + get_file_name(args.claim_scores_path) + ".jsonl")
    
    if os.path.exists(out_put_path):
        logger.info(f"loading predictions from: {out_put_path}")
        fever_result = read_jsonl(out_put_path)
    else:
        fever_result = predict(args)
        logger.info(f"Saving predictions to: {out_put_path}")
        save_jsonl(fever_result, out_put_path)

    if 'evidence' in fever_result[0]:
        data_loose_single, data_strict_single, data_multi_strict_level0, data_multi_strict_level1, data_multi_strict_level2, data_multi_strict_level3 = get_multiHop_and_singleHop_data(fever_result)
        logger.info(f"len(fever_result): {len(fever_result)}")
        logger.info(f"len(data_loose_single): {len(data_loose_single)}")
        logger.info(f"len(data_strict_single): {len(data_strict_single)}")
        logger.info(f"len(data_multi_strict_level0): {len(data_multi_strict_level0)}")
        logger.info(f"len(data_multi_strict_level1): {len(data_multi_strict_level1)}")
        logger.info(f"len(data_multi_strict_level2): {len(data_multi_strict_level2)}")
        logger.info(f"len(data_multi_strict_level3): {len(data_multi_strict_level3)}")
        
        result_type = ['ALL', 'LooseSingleHop', 'StrictSingleHop', 'StrictMultiHop_Level_0', 'StrictMultiHop_Level_1', 'StrictMultiHop_Level_2', 'StrictMultiHop_Level_3']
        me = 5
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
            
            strict_score, acc_score, pr, rec, f1 = fever_score(r,  max_evidence=me)
            logger.info(f"\n******* fever scores @ {me}**********")
            logger.info(f"strict_score: {strict_score}")
            logger.info(f"acc_score: {acc_score}")
            logger.info(f"pr: {pr}")
            logger.info(f"rec: {rec}")
            logger.info(f"f1: {f1}")

            eval_doc_recall_in_fever_format(r, max_evidence=me)

            custom_eval(r, max_evidence=me)

    logger.info("All Done.")
