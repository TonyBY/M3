import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

PATH_TO_SENTEVAL = './SentEval'
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from tqdm import tqdm
import logging
from prettytable import PrettyTable

import torch
from transformers import (AutoTokenizer, DPRContextEncoderTokenizer)

from src.utils.config import parser

from src.models.joint_retrievers import SingleEncoderRetriever, BiEncoderRetriever
from src.utils.model_utils import load_saved_model
from src.utils.data_utils import make_directory
from src.utils.args import prepare_logger

args = parser.parse_args()
logger = logging.getLogger()

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    logger.info(tb)

def batcher(params, 
            batch, 
            max_length=256):
    """
    batcher is a necessary input of the senteval engin.
    It defines moethod that translate a batch of sentences to embeddings.
    
    params:
        batch: a list of sentences.
    """
    # Handle rare token encoding issues in the dataset
    if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        batch = [[word.decode('utf-8') for word in s] for s in batch]

    sentences = [' '.join(s) for s in batch]

    batch = tokenizer.batch_encode_plus(
                                        sentences,
                                        return_tensors="pt",
                                        padding='max_length', 
                                        max_length=max_length, 
                                        truncation=True, 
                                        )

    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)

    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        pooler_output = outputs.pooler_output

    return pooler_output.cpu()

# SentEval prepare and batcher
def prepare(params, samples):
    return
    

if __name__ == '__main__':
    make_directory(args.SE_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.SE_dir, "sentence_encoding_evaluation.log"))
    logger.info(args)

    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

    PATH_TO_DATA = args.sentence_encoding_eval_data_path
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.encoding_batch_size,
                                    'tenacity': 5, 'epoch_size': 4}
    
    if args.single_encoder:
        dsr_model = SingleEncoderRetriever(encoder_dir=args.model_type)
        dsr_model, _ = load_saved_model(dsr_model, args.model_path)

        model = dsr_model.encoder
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    else:
        dsr_model = BiEncoderRetriever(
                                        query_encoder_dir=args.query_encoder_name,
                                        ctx_encoder_dir=args.ctx_encoder_name,
                                    )
        dsr_model, _ = load_saved_model(dsr_model, args.model_path)

        if args.encoder_type == 'ctx':
            model = dsr_model.ctx_encoder
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.ctx_encoder_name)
        elif args.encoder_type == 'query':
            model = dsr_model.query_encoder
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.query_encoder_name)
        else:
            raise Exception(f"Unknow encoder_type: {args.encoder_type}. Only Support 'ctx' and 'query'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = {}

    for task in tqdm(tasks):
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    logger.info(f"------ {args.run_name} eval results ------")

    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

    task_names = []
    scores = []
    for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        task_names.append(task)
        if task in results:
            scores.append("%.2f" % (results[task]['acc']))    
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)
