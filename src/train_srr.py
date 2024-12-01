from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
sys.path.append(pwd)

import math
import logging
import pandas as pd
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

import transformers
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

from src.utils.config import parser
from src.utils.args import prepare_logger
from src.utils.data_utils import load_pickle, make_directory
from src.models.roberta_model import RoBERTa
from src.data.sentence_classification_dataset import SentenceDatasetRoBERTa, collate_fn

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()


def train(args):
    binary_labels = True if args.num_labels == 2 else False
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=args.reranking_dir,
        filename=f"ss_{args.model_type}_binary_{binary_labels}"
        + "_{epoch:02d}-{valid_accuracy:.5f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_type, add_prefix_space=True)

    if args.reranking_train_file.endswith('.pkl'):
        train_csv_dataset = load_pickle(args.reranking_train_file)
    else:
        train_csv_dataset = pd.read_csv(args.reranking_train_file, index_col=0).values

    if args.reranking_dev_file.endswith('.pkl'):
        dev_csv_dataset = load_pickle(args.reranking_dev_file)
    else:
        dev_csv_dataset = pd.read_csv(args.reranking_dev_file, index_col=0).values

    train_dataset = SentenceDatasetRoBERTa(
        train_csv_dataset,
        tokenizer,
        binary_label=binary_labels,
        claim_second=True,
        weighted_sampling=args.weighted_sampling,
    )

    dev_dataset = SentenceDatasetRoBERTa(
        dev_csv_dataset,
        tokenizer,
        binary_label=binary_labels,
        claim_second=True,
    )
    partial_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_length=args.max_seq_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.retrank_batch_size,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        shuffle=True if not args.weighted_sampling else False,
        sampler= train_dataset.sampler,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=2 * args.retrank_batch_size,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
    )
    steps_per_epoch = math.ceil(
        (len(train_dataset) / args.retrank_batch_size) / args.accumulate_gradients
    )

    logger.info(f"args.num_train_epochs: {args.num_train_epochs}")
    if args.init_checkpoint:
        logger.info(f"args.model_type: {args.model_type}")
        logger.info(f"args.model_path: {args.model_path}")

        ckpt = torch.load(args.model_path)
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            steps_per_epoch=steps_per_epoch,
            epochs=args.num_train_epochs,
            class_weights=train_dataset.nli_class_weights,
            use_weighted_ce_loss=args.use_weighted_ce_loss,
            lr=args.learning_rate,
        )
        ss_roberta.load_state_dict(ckpt["state_dict"])
    else: 
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            steps_per_epoch=steps_per_epoch,
            epochs=args.num_train_epochs,
            class_weights=train_dataset.nli_class_weights,
            use_weighted_ce_loss=args.use_weighted_ce_loss,
            lr=args.learning_rate,
        )
    trainer = pl.Trainer(
                        accelerator='auto',
                        max_epochs=args.num_train_epochs,
                        precision="bf16",
                        callbacks=[lr_callback, 
                                   checkpoint_callback],
                        accumulate_grad_batches=args.accumulate_gradients,
                        val_check_interval=0.1,
                        enable_progress_bar=True,
                )
    
    logger.info(f"ss_roberta.steps_per_epoch: {ss_roberta.steps_per_epoch}")
    trainer.fit(
        ss_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )

if __name__ == "__main__":
    make_directory(args.reranking_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.reranking_dir, "reranker_training.log"))
    logger.info(args)
    train(args)
