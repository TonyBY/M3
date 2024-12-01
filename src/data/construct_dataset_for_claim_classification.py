import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from tqdm import tqdm
import time
import logging

from src.utils.data_utils import get_sentence_by_id, process_evid, load_pickle, dump_pickle, get_label_num, make_directory, get_file_name
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

def main(args, sep="|#SEP#|"):
    start = time.time()

    logger.info(f"Loading final_retrieval_results from: {args.final_retrieval_results_path}")
    final_retrieval_results = load_pickle(args.final_retrieval_results_path)

    if args.debug:
        final_retrieval_results = final_retrieval_results[:100]

    logger.info(f"Loading wiki_line_dict from: {args.wiki_line_dict_pkl_path}")
    wiki_line_dict = load_pickle(args.wiki_line_dict_pkl_path)

    data = []
    for item in tqdm(final_retrieval_results):
        claim = item['claim']
        label = item['label']

        pred_evidence_list = item[args.retrieved_evidence_feild]
        logger.debug(f"len(pred_evidence_list): {len(pred_evidence_list)}")
        logger.debug(f"pred_evidence_list: {pred_evidence_list}")

        lines = []
        for evi in pred_evidence_list[:5]:
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

        if lines == []:
            logger.info(f"WARNING: No evidence is available.")
            continue

        sample = {"claim": claim, 
                  "pred_evi": lines, 
                  "label": get_label_num(label)
                }
        logger.debug(f"sample: {sample}")
        data.append(sample)

    end = time.time()
    total_time = end - start
    logger.info(f"total_time: {total_time}")
    
    if args.debug:
        dump_pickle(data, os.path.join(args.claim_classification_dir, 'DEBUG_claim_classification_' + get_file_name(args.final_retrieval_results_path) + '.pkl'))
    else:
        dump_pickle(data, os.path.join(args.claim_classification_dir, 'claim_classification_' + get_file_name(args.final_retrieval_results_path) + '.pkl'))


if __name__ == "__main__":
    make_directory(args.claim_classification_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.claim_classification_dir, 'claim_classification_' + get_file_name(args.final_retrieval_results_path) + '.log'))
    logger.info(args)
    main(args)
  