# Claim Classifier Training
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-1]))
sys.path.append(pwd)

import math
import logging
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
from src.data.claim_classification_dataset import ClaimClassificationDataset, collate_fn

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

def train(args):
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=args.claim_classification_dir,
        filename=f'cc_{args.model_type}_concat_'
        + "{epoch:02d}-{valid_accuracy:.5f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, add_prefix_space=True)

    train_data = load_pickle(args.claim_classificaton_train_file)
    dev_data = load_pickle(args.claim_classificaton_dev_file)

    train_dataset = ClaimClassificationDataset(
        train_data,
        train=True,
        shuffle_evidence_p=args.shuffle_evidence_p,
        weighted_sampling=args.weighted_sampling,
    )

    dev_dataset = ClaimClassificationDataset(
        dev_data,
        train=False,
        shuffle_evidence_p=args.shuffle_evidence_p,
        weighted_sampling=False,
    )

    partial_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_length=args.max_seq_len
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.cc_batch_size,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        shuffle=True if not args.weighted_sampling else False,
        sampler= train_dataset.sampler,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=2 * args.cc_batch_size,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(
        (len(train_dataset) / args.cc_batch_size) / args.accumulate_gradients
    )

    logger.info(f"args.num_train_epochs: {args.num_train_epochs}")
    logger.info(f"args.model_type: {args.model_type}")
    cc_roberta = RoBERTa(
            model_type=args.model_type,
            num_labels=3,
            steps_per_epoch=steps_per_epoch,
            epochs=args.num_train_epochs,
            class_weights=train_dataset.nli_class_weights,
            use_weighted_ce_loss=args.use_weighted_ce_loss,
            label_smoothing=args.label_smoothing,
            lr=args.learning_rate,
        )
    # cc_roberta.expand_embeddings()
    
    if args.init_checkpoint:
        logger.info(f"initiating model parameters from:: {args.model_path}")
        
        ckpt = torch.load(args.model_path)
        cc_roberta.load_state_dict(ckpt["state_dict"], strict=False)

    cc_roberta.train()
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=args.num_train_epochs,
        default_root_dir="checkpoints",
        precision="bf16",
        callbacks=[lr_callback, checkpoint_callback],
        accumulate_grad_batches=args.accumulate_gradients,
        val_check_interval=args.val_check_interval,
        strategy="ddp",
    )
    trainer.fit(
        cc_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )


if __name__ == "__main__":
    make_directory(args.claim_classification_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.claim_classification_dir, "claim_classifier_training.log"))
    logger.info(args)
    train(args)
    