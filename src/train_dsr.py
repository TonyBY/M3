from __future__ import absolute_import
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

import random
import logging
import numpy as np
from tqdm import tqdm
from datetime import date
from typing import List

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

import transformers
from transformers import (AutoTokenizer, DPRQuestionEncoderTokenizer, 
                          DPRContextEncoderTokenizer, get_linear_schedule_with_warmup)

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.config import parser
from src.utils.args import check_cuda, prepare_logger
from src.utils.model_utils import load_saved_model, AverageMeter, get_optimizer, save_model
from src.utils.data_utils import move_to_device, get_attn_mask
from src.models.joint_retrievers import BiEncoderRetriever, SingleEncoderRetriever
from src.models.criterions import multi_task_loss, batch_eval
from src.data.joint_dsr_dataset import JointDSRDataset, BiEncoderBatch, batch_collate

transformers.logging.set_verbosity_error()
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

args = parser.parse_args()
logger = logging.getLogger()

def main(args):
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}-negsize{args.num_hard_negs}-shared{args.shared_encoder}-mode{args.train_mode}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.num_train_epochs != 0 and not args.debug:
        raise Exception(f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs","tflogs")))

    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.output_dir, "training.log"))
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.single_encoder:
        model = SingleEncoderRetriever(encoder_dir=args.ctx_encoder_name,
                                       n_classes=3)
        max_position_embeddings = model.encoder.config.max_position_embeddings
    else:
        model = BiEncoderRetriever(query_encoder_dir=args.query_encoder_name,
                                    ctx_encoder_dir=args.ctx_encoder_name,
                                    n_classes=3,
                                    share_encoder=args.shared_encoder)
        max_position_embeddings = model.ctx_encoder.config.max_position_embeddings

    hyper_params_cache = {}
    if args.init_checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        model, hyper_params_cache = load_saved_model(model, args.checkpoint_path)

    logger.info("=========================================")
    if not args.single_encoder and args.freeze_ctx_encoder:
        for name, param in model.ctx_encoder.named_parameters():
            logger.info(f"Freezing parameters of layer: {name}")
            logger.info(f"name: {name}")
            param.requires_grad = False

    model.to(device)

    optimizer = (
            get_optimizer(
                            model,
                            learning_rate=args.learning_rate,
                            adam_eps=args.adam_epsilon,
                            weight_decay=args.weight_decay,
                        )
            if args.do_train
            else None
        )

    if 'dpr' in args.ctx_encoder_name.lower():
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.ctx_encoder_name)
    else:
        ctx_tokenizer = AutoTokenizer.from_pretrained(args.ctx_encoder_name)

    if args.shared_encoder or args.single_encoder:
        query_tokenizer = None
    else:
        if 'dpr' in args.query_encoder_name.lower():
            query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.query_encoder_name)
        else:
            query_tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_name)

        assert query_tokenizer.pad_token_id == ctx_tokenizer.pad_token_id

    pad_token_id = ctx_tokenizer.pad_token_id
        
    if args.do_train and args.max_seq_len > max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, max_position_embeddings))
    elif args.do_train and args.max_seq_len > (max_position_embeddings / 2) and (args.train_mode == 'JOINT' or args.train_mode == 'NLI_ONLY'):
        raise ValueError(
            f"Cannot use sequence length {args.max_c_len} under training mode: {args.train_mode}. Maximum supported sequence length {(max_position_embeddings) / 2}")

    logger.info("Preparing eval_dataset ...")
    eval_dataset = JointDSRDataset(query_tokenizer=query_tokenizer, 
                                    ctx_tokenizer=ctx_tokenizer, 
                                    data_path=args.development_file, 
                                    max_len=args.max_seq_len, 
                                    num_hard_negs=args.num_hard_negs,
                                    train=False,
                                    train_mode=args.train_mode)

    eval_dataloader = DataLoader(eval_dataset, 
                                batch_size=args.predict_batch_size, 
                                pin_memory=True, 
                                collate_fn=lambda x: x,
                                num_workers=args.num_workers)

    logger.info(f"len(eval_dataset): {len(eval_dataset)}")
    logger.info(f"len(eval_dataloader): {len(eval_dataloader)}")

    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.local_rank != -1 and not args.no_cuda:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        if hyper_params_cache != {} and args.continue_training:
            epoch_start = hyper_params_cache['epoch'] + 1
            global_step = hyper_params_cache['global_step'] # gradient update step
            batch_step = hyper_params_cache['batch_step'] # forward batch count
            best_mrr = hyper_params_cache['best_mrr']
            best_nli_f1 = hyper_params_cache['best_nli_f1']
            train_loss_meter = hyper_params_cache['train_loss_meter']
        else:
            epoch_start = 0
            global_step = 0
            batch_step = 0
            best_mrr = 0
            best_nli_f1 = 0
            train_loss_meter = AverageMeter()

        logger.info(f"epoch_start: {epoch_start}")
        logger.info(f"global_step: {global_step}")
        logger.info(f"batch_step: {batch_step}")
        logger.info(f"best_mrr: {best_mrr}")
        logger.info(f"best_nli_f1: {best_nli_f1}")

        model.train()

        logger.info("Preparing train_dataset ...")
        train_dataset = JointDSRDataset(query_tokenizer=query_tokenizer, 
                                        ctx_tokenizer=ctx_tokenizer, 
                                        data_path=args.train_file, 
                                        max_len=args.max_seq_len, 
                                        num_hard_negs=args.num_hard_negs,
                                        train=True,
                                        train_mode=args.train_mode,
                                        use_weighted_ce_loss=args.use_weighted_ce_loss,
                                        weighted_sampling=args.weighted_sampling,
                                        target_nli_distribution=args.target_nli_distribution,)

        nli_class_weights = train_dataset.nli_class_weights

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=args.train_batch_size, 
                                      pin_memory=True, 
                                      collate_fn=lambda x: x,
                                      num_workers=args.num_workers,
                                      shuffle=True if not args.weighted_sampling else False,
                                      sampler= train_dataset.sampler,
                                      drop_last=True,)
        logger.info(f"len(train_dataset): {len(train_dataset)}")
        logger.info(f"len(train_dataloader): {len(train_dataloader)}")

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        if args.debug:
            args.eval_period = 10
        else:
            args.eval_period = len(train_dataloader) // 5
        logger.info(f"args.eval_period: {args.eval_period}")

        if args.use_extra_retrieve_only_train_dataset:
            logger.info("Preparing extra_retrieve_only_train_dataset ...")
            extra_retrieve_only_train_dataset = JointDSRDataset(query_tokenizer=query_tokenizer, 
                                                                ctx_tokenizer=ctx_tokenizer, 
                                                                data_path=args.extra_retrieve_only_train_file, 
                                                                max_len=args.max_seq_len, 
                                                                num_hard_negs=args.num_hard_negs,
                                                                train=True,
                                                                train_mode="RETRIEVE_ONLY")

            extra_retrieve_only_train_dataloader = DataLoader(extra_retrieve_only_train_dataset, 
                                                              batch_size=args.train_batch_size, 
                                                              pin_memory=True, 
                                                              collate_fn=lambda x: x,
                                                              num_workers=args.num_workers,
                                                              shuffle=True,
                                                              drop_last=True,)
            logger.info(f"len(extra_retrieve_only_train_dataset): {len(extra_retrieve_only_train_dataset)}")
            logger.info(f"len(extra_retrieve_only_train_dataloader): {len(extra_retrieve_only_train_dataloader)}")

            t_total = (len(train_dataloader) + len(extra_retrieve_only_train_dataloader)) // args.gradient_accumulation_steps * args.num_train_epochs
        
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        logger.info('Start training....')
        
        if device.type == 'cuda':
            scaler = GradScaler()
        else:
            scaler = None
            logger.info("Not using loss scaler.")

        stop_training = False

        eval_scores = predict(args, model.module if n_gpu > 1 and not args.no_cuda else model, eval_dataloader, device, logger, pad_token_id, train_mode=args.train_mode)
        mrr = eval_scores["mrr"]
        nli_f1 = eval_scores["nli_f1"]
        joint_score = mrr + nli_f1
        logger.info(f"Step {global_step} Train loss.avg {train_loss_meter.avg} MRR{mrr*100} nli_f1 {nli_f1*100} on epoch={-1}")
        # exit()

        for epoch in range(epoch_start, epoch_start + int(args.num_train_epochs)):
            logger.info(f"epoch: {epoch} >>>>>>>>>>")
            if args.use_extra_retrieve_only_train_dataset and args.mix_interval != 0:
                if epoch % args.mix_interval == 0:
                    batch_step, global_step, train_loss_meter, best_mrr, best_nli_f1, stop_training = trainer(args, model, extra_retrieve_only_train_dataloader, eval_dataloader, tb_logger, 
                                                                                                                epoch=epoch,
                                                                                                                batch_step=batch_step,
                                                                                                                global_step=global_step,
                                                                                                                best_mrr=best_mrr,
                                                                                                                best_nli_f1=best_nli_f1, 
                                                                                                                device=device,
                                                                                                                pad_token_id=pad_token_id,
                                                                                                                n_gpu=n_gpu,
                                                                                                                scaler=scaler,
                                                                                                                optimizer=optimizer,
                                                                                                                scheduler=scheduler,
                                                                                                                train_loss_meter=train_loss_meter,
                                                                                                                train_mode="RETRIEVE_ONLY",
                                                                                                                do_eval=False,)
                    
            batch_step, global_step, train_loss_meter, best_mrr, best_nli_f1, stop_training = trainer(args, model, train_dataloader, eval_dataloader, tb_logger, 
                                                                                                        epoch=epoch,
                                                                                                        batch_step=batch_step,
                                                                                                        global_step=global_step,
                                                                                                        best_mrr=best_mrr,
                                                                                                        best_nli_f1=best_nli_f1,
                                                                                                        device=device,
                                                                                                        pad_token_id=pad_token_id,
                                                                                                        n_gpu=n_gpu,
                                                                                                        scaler=scaler,
                                                                                                        optimizer=optimizer,
                                                                                                        scheduler=scheduler,
                                                                                                        train_loss_meter=train_loss_meter,
                                                                                                        train_mode=args.train_mode,
                                                                                                        nli_class_weights=nli_class_weights, 
                                                                                                        do_eval=True,)
                                                                                                
            eval_scores = predict(args, model.module if n_gpu > 1 and not args.no_cuda else model, eval_dataloader, device, logger, pad_token_id, train_mode=args.train_mode)
            mrr = eval_scores["mrr"]
            nli_f1 = eval_scores["nli_f1"]
            joint_score = mrr + nli_f1
            logger.info(f"Step {global_step} Train loss.avg {train_loss_meter.avg} MRR{mrr*100} nli_f1 {nli_f1*100} on epoch={epoch}")
            for k, v in eval_scores.items():
                tb_logger.add_scalar(k, v*100, epoch)
            save_model(args, model.module if hasattr(model, 'module') else model,
                        epoch=epoch,
                        global_step=global_step, 
                        batch_step=batch_step,
                        best_mrr=best_mrr,
                        best_nli_f1=best_nli_f1,
                        train_loss_meter=train_loss_meter,
                        checkpoint_name='checkpoint_last')

            if best_nli_f1 < nli_f1:
                pre_best_nli_f1 = best_nli_f1
                best_nli_f1 = nli_f1                    
                if args.train_nli_only or (args.use_nli_best_score and not args.use_joint_best_score):
                    logger.info(f"Saving model with best nli_f1 {pre_best_nli_f1*100} -> best_nli_f1 {best_nli_f1*100} on epoch={epoch}")
                    save_model(args, model.module if hasattr(model, 'module') else model,
                            epoch=epoch,
                            global_step=global_step, 
                            batch_step=batch_step,
                            best_mrr=best_mrr,
                            best_nli_f1=best_nli_f1,
                            train_loss_meter=train_loss_meter)

            if best_mrr < mrr:
                pre_best_mrr = best_mrr
                best_mrr = mrr
                if not args.use_joint_best_score and not args.use_nli_best_score and not args.train_nli_only:
                    logger.info(f"Saving model with best MRR {pre_best_mrr*100} -> MRR {best_mrr*100} on epoch={epoch}")
                    save_model(args, model.module if hasattr(model, 'module') else model,
                                epoch=epoch,
                                global_step=global_step, 
                                batch_step=batch_step,
                                best_mrr=best_mrr,
                                best_nli_f1=best_nli_f1,
                                train_loss_meter=train_loss_meter)

            best_joint_score = best_mrr + best_nli_f1
            if best_joint_score < joint_score:
                pre_best_joint_score = best_joint_score
                best_joint_score = joint_score
                if args.use_joint_best_score and not args.train_nli_only:
                    logger.info(f"Saving model with best JOINT_SCORE {pre_best_joint_score*100} -> JOINT_SCORE {best_joint_score*100} on epoch={epoch}")
                    save_model(args, model.module if hasattr(model, 'module') else model,
                                epoch=epoch,
                                global_step=global_step, 
                                batch_step=batch_step,
                                best_mrr=best_mrr,
                                best_nli_f1=best_nli_f1,
                                train_loss_meter=train_loss_meter)

            if stop_training:
                break

        logger.info(f"Training finished! Total epochs trained: {args.num_train_epochs}")

    elif args.do_predict:
        acc = predict(args, model.module if n_gpu > 1 and not args.no_cuda else model, eval_dataloader, device, logger, pad_token_id, train_mode=args.train_mode)
        logger.info(f"test performance {acc}")

def trainer(args, model, train_data_loader: DataLoader, eval_dataloader: DataLoader, tb_logger: SummaryWriter,
            epoch: int=None,
            batch_step: int=None,
            global_step: int=None,
            best_mrr: float=None,
            best_nli_f1: float=None, 
            device=None, 
            pad_token_id=None,
            n_gpu: int=None,
            scaler=None,
            optimizer=None,
            scheduler=None,
            train_loss_meter: AverageMeter=None,
            train_mode: str="RETRIEVE_ONLY", 
            nli_class_weights: List[int]=None,
            do_eval: bool=True,
            ):

    stop_training = False
    best_joint_score = best_mrr + best_nli_f1

    i = 0
    for batch in tqdm(train_data_loader):
        batch_step += 1
        batch = batch_collate(batch, train_mode=train_mode, num_hard_negs=args.num_hard_negs)
        batch = BiEncoderBatch(**move_to_device(batch._asdict(), device))
        
        q_attn_mask = get_attn_mask(pad_token_id, batch.question_ids)
        ctx_attn_mask = get_attn_mask(pad_token_id, batch.context_ids)

        with autocast():
            output = model(
                            batch.question_ids,
                            q_attn_mask,
                            batch.question_segments,
                            batch.context_ids,
                            ctx_attn_mask,
                            batch.ctx_segments,
                            batch.is_positive_mask,
                            batch.nli_labels,
                            train_mode=train_mode
                        )

            if args.positive_weight <= 0 or args.positive_weight >= 1:
                positive_weight = None
            else:
                positive_weight = args.positive_weight
            con_loss, con_correct_num, nli_loss, nli_correct_num, loss = multi_task_loss(batch, output, 
                                                                                         train_mode=train_mode, 
                                                                                         positive_weight=positive_weight,
                                                                                         nli_class_weights=nli_class_weights,
                                                                                         retrieval_to_nli_weight=args.retrieval_to_nli_weight,
                                                                                         sim_type=args.sim_type, 
                                                                                         temp=args.temp,
                                                                                         CROSSENTROPY_LOSS=args.use_ce_loss,
                                                                                         )

            logger.info(f"con_loss: {con_loss}, con_correct_num: {con_correct_num}, nli_loss: {nli_loss}, nli_correct_num: {nli_correct_num}, joint_loss: {loss}")
            if train_mode == "RETRIEVE_ONLY":
                loss = con_loss
                logger.info(f"RETRIEVE_ONLY mode, loss --> {loss}")
            elif train_mode == "NLI_ONLY":
                loss = nli_loss
                logger.info(f"NLI_ONLY mode, loss --> {loss}")
            elif train_mode == "JOINT":
                logger.info(f"JOINT mode, loss --> {loss}")
            else:
                raise Exception(f"Unknown train_mode: {train_mode}. Supported train_mode: {['RETRIEVE_ONLY', 'NLI_ONLY', 'JOINT']}")

        if n_gpu > 1 and not args.no_cuda:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if torch.isnan(loss).data:
            logger.info("Stop training because loss=%s" % (loss.data))
            stop_training = True
            break
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
            logger.debug(f"loss_scale: {scaler.get_scale()}")
        else:
            loss.backward()

        
        train_loss_meter.update(loss.item())
    
        if (batch_step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            model.zero_grad()
            global_step += 1

            tb_logger.add_scalar('batch_train_loss',
                                loss.item(), global_step)
            tb_logger.add_scalar('smoothed_train_loss',
                                train_loss_meter.avg, global_step)

            logger.debug(f"global_step: {global_step}")
            logger.debug(f"args.eval_period: {args.eval_period}")
            logger.debug(f"global_step % args.eval_period: {global_step % args.eval_period}")
            if do_eval and args.eval_period != -1 and global_step % args.eval_period == 0:
                eval_scores = predict(args, model.module if n_gpu > 1 and not args.no_cuda else model, eval_dataloader, device, logger, pad_token_id, train_mode=args.train_mode)
                mrr = eval_scores["mrr"]
                nli_f1 = eval_scores["nli_f1"]
                joint_score = mrr + nli_f1

                logger.info(f"Step {global_step} Train loss {train_loss_meter.avg} MRR: {mrr*100}, nli_f1: {nli_f1*100}, joint_score: {joint_score*100}, on epoch={epoch}")
                tb_logger.add_scalar('dev_batch_mrr',
                                        mrr*100.0, global_step) 

                if best_nli_f1 < nli_f1:
                    pre_best_nli_f1 = best_nli_f1
                    best_nli_f1 = nli_f1                    
                    if args.train_nli_only or (args.use_nli_best_score and not args.use_joint_best_score):
                        logger.info(f"Saving model with best nli_f1 {pre_best_nli_f1*100} -> best_nli_f1 {best_nli_f1*100} on epoch={epoch}")
                        save_model(args, model.module if hasattr(model, 'module') else model,
                                epoch=epoch,
                                global_step=global_step, 
                                batch_step=batch_step,
                                best_mrr=best_mrr,
                                best_nli_f1=best_nli_f1,
                                train_loss_meter=train_loss_meter)

                if best_mrr < mrr:
                    pre_best_mrr = best_mrr
                    best_mrr = mrr
                    if not args.use_joint_best_score and not args.use_nli_best_score and not args.train_nli_only:
                        logger.info(f"Saving model with best MRR {pre_best_mrr*100} -> MRR {best_mrr*100} on epoch={epoch}")
                        save_model(args, model.module if hasattr(model, 'module') else model,
                                    epoch=epoch,
                                    global_step=global_step, 
                                    batch_step=batch_step,
                                    best_mrr=best_mrr,
                                    best_nli_f1=best_nli_f1,
                                    train_loss_meter=train_loss_meter)
                        # model = model.to(device)

                if best_joint_score < joint_score:
                    pre_best_joint_score = best_joint_score
                    best_joint_score = joint_score
                    if args.use_joint_best_score and not args.train_nli_only:
                        logger.info(f"Saving model with best JOINT_SCORE {pre_best_joint_score*100} -> JOINT_SCORE {best_joint_score*100} on epoch={epoch}")
                        save_model(args, model.module if hasattr(model, 'module') else model,
                                    epoch=epoch,
                                    global_step=global_step, 
                                    batch_step=batch_step,
                                    best_mrr=best_mrr,
                                    best_nli_f1=best_nli_f1,
                                    train_loss_meter=train_loss_meter)
    
    return batch_step, global_step, train_loss_meter, best_mrr, best_nli_f1, stop_training

def predict(args, model, eval_dataloader, device, logger, pad_token_id, train_mode:str='JOINT'):
    model.eval()
    rrs = [] # reciprocal rank
    nli_preds = []
    nli_labels = []
    i = 0
    for batch in tqdm(eval_dataloader):
        # if i >= 20:
        #     break
        # i += 1
        batch = batch_collate(batch, train_mode=train_mode, num_hard_negs=args.num_hard_negs)
        batch = BiEncoderBatch(**move_to_device(batch._asdict(), device))

        q_attn_mask = get_attn_mask(pad_token_id, batch.question_ids)
        ctx_attn_mask = get_attn_mask(pad_token_id, batch.context_ids)

        with torch.no_grad(), autocast():
            output = model(
                            batch.question_ids,
                            q_attn_mask,
                            batch.question_segments,
                            batch.context_ids,
                            ctx_attn_mask,
                            batch.ctx_segments,
                            batch.is_positive_mask,
                            batch.nli_labels,
                            train_mode=train_mode,
                        )
            eval_scores = batch_eval(batch, output)
            rrs += eval_scores['rrs']
            nli_preds += eval_scores['nli_preds']
            nli_labels += eval_scores['nli_labels']

    mrr = np.mean(rrs) if rrs != [] else 0



    logger.info(f"evaluated {len(nli_labels)} examples...")
    logger.info(f'MRR: {mrr}')

    if train_mode == 'JOINT' or train_mode == 'NLI_ONLY':
        nli_f1 = f1_score(nli_labels, nli_preds, average='macro', zero_division=0)
        nli_precision = precision_score(nli_labels, nli_preds, average='macro', zero_division=0)
        nli_recall = recall_score(nli_labels, nli_preds, average='macro', zero_division=0)
        
        try:
            logger.info(f"nli_report: \n{classification_report(nli_labels, nli_preds, zero_division=0, target_names = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'])}")
        except ValueError:
            logger.info(f"nli_report: \n{classification_report(nli_labels, nli_preds, zero_division=0, target_names = ['SUPPORTS', 'REFUTES'])}")

        logger.info(f"nli_confusion_matrix: \n{confusion_matrix(nli_labels, nli_preds)}")

    else:
        nli_f1 = 0
        nli_precision = 0
        nli_recall = 0

    logger.info(f'nli_f1: {nli_f1*100}')
    logger.info(f'nli_recall: {nli_recall*100}')
    logger.info(f'nli_precision: {nli_precision*100}')

    model.train()
    return {"mrr": mrr, "nli_f1": nli_f1, "nli_recall": nli_recall, "nli_precision": nli_precision}


if __name__ == "__main__":
    check_cuda(args.use_cuda)
    print(f"args.use_extra_retrieve_only_train_dataset: {args.use_extra_retrieve_only_train_dataset}")
    main(args)
