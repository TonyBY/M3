from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import torch
from torch import nn
from typing import List
import os
from src.utils.data_utils import dump_pickle, load_pickle, get_file_dir, get_file_name

def load_saved_model(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    def filter(x): return x[7:] if x.startswith('module.') else x
    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)

    model_dir = get_file_dir(path)
    check_point_name = get_file_name(path)
    if os.path.join(model_dir, f"{check_point_name}_args_and_step.pkl"):
        hyper_params = load_pickle(os.path.join(model_dir, f"{check_point_name}_args_and_step.pkl"))
    else:
        hyper_params = {}
    return model, hyper_params

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(args, model, 
               epoch: int=None,
               global_step: int=None, 
               batch_step: int=None, 
               best_mrr: float=None, 
               best_nli_f1: float=None,
               best_stsb: float=None,
               best_stskr: float=None,
               train_loss_meter:AverageMeter=None,
               checkpoint_name: str='checkpoint_best',
               task: str='dsr'):
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{checkpoint_name}.pt"))
    if task == 'dsr':
        dump_pickle({"args": args, 
                    "epoch": epoch,
                    "global_step": global_step, 
                    "batch_step": batch_step, 
                    "best_mrr": best_mrr, 
                    "best_nli_f1": best_nli_f1,
                    "train_loss_meter": train_loss_meter}, 
                    os.path.join(args.output_dir, f"{checkpoint_name}_args_and_step.pkl"))
    elif task == 'use':
        dump_pickle({"args": args, 
                    "epoch": epoch,
                    "global_step": global_step, 
                    "batch_step": batch_step, 
                    "best_stsb": best_stsb, 
                    "best_stskr": best_stskr,
                    "train_loss_meter": train_loss_meter}, 
                    os.path.join(args.output_dir, f"{checkpoint_name}_args_and_step.pkl"))
    else:
        raise Exception(f"Unknow task type: {task}.")

def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay=weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate=learning_rate, adam_eps=adam_eps)
    