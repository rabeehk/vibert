""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import argparse
import glob
import logging
import os
import csv
import random
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from prior_wd_optim import PriorWD

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from models import BertForSequenceClassification
from utils import write_to_csv
from data import convert_examples_to_features, processors, output_modes, glue_compute_metrics


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    stop_training = 0
    def save_model(args, global_step, model, optimizer, scheduler, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.prior_weight_decay: # I am just addding this because revisiting bert few-sample added it. should be checked.
       optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,\
            correct_bias=not args.use_bertadam, weight_decay=args.weight_decay)
    else:
       optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.prior_weight_decay:
       optimizer = PriorWD(optimizer, use_prior_wd=args.prior_weight_decay) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,\
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        if stop_training == 2:
            break;
        epoch = epoch + 1
        args.epoch = epoch
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs, epoch=epoch)
            loss = outputs["loss"]["loss"]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _, _ = evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, model, optimizer, scheduler, tokenizer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # Evaluates the model after each epoch.
        if args.evaluate_after_each_epoch:
            results, _, _, _ = evaluate(args, model, tokenizer, epoch=epoch)
            #save_model(args, global_step, model, optimizer, scheduler, tokenizer)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def binarize_preds(preds):
    # maps the third label (neutral one) to first, which is contradiction.
    preds[preds == 2] = 0
    return preds

def compute_metrics(args, task, preds, out_label_ids):
    return glue_compute_metrics(task, preds, out_label_ids)

def evaluate(args, model, tokenizer, prefix="", sampling_type="argmax", save_results=True, epoch=0):
    results = {}
    all_preds = {}
    all_zs = {}
    all_labels = {}
    for eval_task in args.eval_tasks:
        for eval_type in args.eval_types:
            print("Evaluating on "+eval_task+" with eval_type ", eval_type)
            eval_dataset, num_classes = load_and_cache_examples(args, eval_task, tokenizer, eval_type)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            zs = []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                    no_label = True if (eval_type == "test" and eval_task in args.glue_tasks) else False
                    if no_label:
                        inputs["labels"] = None
                    outputs = model(**inputs, sampling_type=sampling_type)
                    tmp_eval_loss, logits = outputs["loss"], outputs["logits"]

                    #eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = None if no_label else inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = None if no_label else np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

                zs.append(outputs["z"])

            #eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)

                # binarize the labels and predictions if needed.
                if num_classes == 2 and args.binarize_eval:
                    preds = binarize_preds(preds)
                    out_label_ids = binarize_preds(out_label_ids)

            elif args.output_mode == "regression":
                preds = np.squeeze(preds)

            all_preds[eval_task + "_" + eval_type] = preds
            all_zs[eval_task+"_"+eval_type] = torch.cat(zs)
            all_labels[eval_task+"_"+eval_type] = out_label_ids

            no_label = True if (eval_type == "test" and eval_task in args.glue_tasks) else False
            if not no_label:
                temp = compute_metrics(args, eval_task, preds, out_label_ids)
                if len(args.eval_tasks) > 1:
                    # then this is for transfer and we need to know the name of the datasets.
                    temp = {eval_task+"_"+k + '_' + eval_type: v for k, v in temp.items()}
                else:
                    temp = {k + '_' + eval_type: v for k, v in temp.items()}
                results.update(temp)
                print(results)
            else:
                write_in_glue_format(args, all_preds, eval_type, epoch=epoch)

    # In case of glue, results is empty.
    if args.outputfile is not None and save_results and results:
        write_to_csv(results, args, args.outputfile)
    return results, all_preds, all_zs, all_labels


def load_and_cache_examples(args, task, tokenizer, eval_type):
    data_dir = args.task_to_data_dir[task]
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if task in args.nli_tasks:
        processor = processors[task](args.task_to_data_dir[task])
    else:
        processor = processors[task]()

    output_mode = output_modes[task]

    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]


    if eval_type == "train":
        if args.sample_train:
            cached_features_file = join(data_dir, 'cached_type_{}_task_{}_sample_train_{}_num_samples_{}_model_{}_data_seed_{}'.\
                                format(eval_type, task, args.sample_train, args.num_samples,
                                       list(filter(None, args.model_name_or_path.split('/'))).pop(), args.data_seed))
        else:
            # here data_seed has no impact.
            cached_features_file = join(data_dir,
                                        'cached_type_{}_task_{}_sample_train_{}_num_samples_{}_model_{}'. \
                                        format(eval_type, task, args.sample_train, args.num_samples,
                                               list(filter(None, args.model_name_or_path.split('/'))).pop()))
    else:
        cached_features_file = join(data_dir, 'cached_type_{}_task_{}_model_{}'. \
                                    format(eval_type, task, list(filter(None, args.model_name_or_path.split('/'))).pop()))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        if eval_type == "train":
            if args.sample_train:
                data_dir = join(data_dir, "sampled_datasets", "seed_"+str(args.data_seed), str(args.num_samples)) # sampled: for old version.
            examples = (processor.get_train_examples(data_dir))
        elif eval_type == "test":
            examples = (processor.get_dev_examples(data_dir))
        elif eval_type == "dev":
            examples = (processor.get_validation_examples(data_dir))

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            output_mode=output_mode,
            no_label=True if (eval_type == "test" and task in args.glue_tasks) else False
        )
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    if args.local_rank == 0 and eval_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset, processor.num_classes

def write_in_glue_format(args, preds, eval_type, epoch):
    def label_from_example(label, output_mode, label_map):
        if output_mode == "classification":
            return label_map[label]
        elif output_mode == "regression":
            return float(label)
        raise KeyError(output_mode)

    def write_labels(labels, outpath):
        with open(outpath, 'wt') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['index', 'prediction'])
            for i, label in enumerate(labels):
                tsv_writer.writerow([i, label])

    task_to_filename={"rte": "RTE", "sts-b":"STS-B", "mrpc": "MRPC"}
    task = args.eval_tasks[0]
    preds = preds[task+"_"+eval_type]
    processor = processors[task]()
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list)}
    output_mode = output_modes[task]
    labels = [label_from_example(label, output_mode, label_map) for label in preds]
    write_labels(labels, join(args.output_dir, task_to_filename[task]+"_"+eval_type+"_epoch_"+str(epoch)+".tsv"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=None, help="dropout rate.")
    parser.add_argument("--mixout", type=float, default=0.0, help="mixout probability (default: 0)")
    parser.add_argument(
        "--prior_weight_decay", action="store_true", help="Weight Decaying toward the bert params",
    )
    parser.add_argument("--kl_annealing", choices=[None, "linear"], default=None)
    parser.add_argument("--sample_train", action="store_true", help="Sample the training set.")
    parser.add_argument("--num_samples", type=int, \
                        help="Defines the number of the training samples.")
    parser.add_argument("--evaluate_after_each_epoch", action="store_true", help="Eveluates the model after\
            each epoch and saves the best model.")
    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
            through mlp in a deterministic manner.")
    parser.add_argument("--activation", type=str, choices=["tanh", "sigmoid", "relu"], \
                        default="relu")
    parser.add_argument("--eval_types", nargs="+", type=str, default=["train", "test"], \
                        choices=["train", "test", "dev"], help="Specifies the types to evaluate on,\
                            can be dev, test, train.")
    parser.add_argument("--binarize_eval", action="store_true", help="If specified, binarize the predictions, and\
            labels during the evaluations in case of binary-class datasets.")
    # Ib parameters.
    parser.add_argument("--beta", type=float, default=1.0, help="Defines the weight for the information bottleneck\
            loss.")
    parser.add_argument("--ib", action="store_true", help="If specified, uses the information bottleneck to reduce\
            the dimensions.")
    parser.add_argument("--sample_size", type=int, default=5, help="Defines the number of samples for the ib method.")
    parser.add_argument("--ib_dim", default=128, type=int,
                        help="Specifies the dimension of the information bottleneck.")

    # Required parameter
    parser.add_argument("--outputfile", type=str, default=None)
    parser.add_argument("--eval_tasks", nargs="+", default=[], type=str, help="Specifies a list of evaluation tasks.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--data_seed", type=int, default=66, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    if args.evaluate_after_each_epoch and "dev" not in args.eval_types:
        args.eval_types.append("dev")

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    args.task_name_to_eval_metric = {
        "wnli": "acc",
        "cola": "mcc",
        "rte": "acc",
        "mrpc": "acc",
        "sts-b": "pearson",
        "snli": "acc"
    }
    args.task_to_data_dir = {
        "rte": "data/datasets/RTE",
        "mrpc": "data/datasets/MRPC",
        "sts-b": "data/datasets/STS-B",
        "yelp": "data/datasets/yelp/",
        "snli": "data/datasets/SNLI",
        "mnli": "data/datasets/MNLI",
        "imdb": "data/datasets/imdb/",
        "mnli-mm": "data/datasets/MNLI",
        "addonerte": "data/datasets/AddOneRTE",
        "dpr": "data/datasets/DPR/",
        "spr": "data/datasets/SPR/",
        "fnplus": "data/datasets/FNPLUS/",
        "joci": "data/datasets/JOCI/",
        "mpe": "data/datasets/MPE/",
        "scitail": "data/datasets/SciTail/",
        "sick": "data/datasets/SICK/",
        "QQP": "data/datasets/QQP/",
        "snlihard": "data/datasets/SNLIHard/"
    }
    args.glue_tasks = ["rte", "mrpc", "sts-b"]
    args.nli_tasks = ["addonerte", "dpr", "spr", "fnplus", "joci", "mpe", "scitail", \
                "sick", "QQP", "snlihard"]

    # Prepare GLUE task
    if len(args.eval_tasks) == 0:
        args.eval_tasks = [args.task_name]

    args.task_name = args.task_name.lower()
    return args


def main():
    args = get_args()

    if args.task_name in args.nli_tasks:
        processor = processors[args.task_name](args.task_to_data_dir[args.task_name])
    else:
        processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type in ["bert", "roberta"]:
       # bert dim is 768.
       args.hidden_dim = (768 + args.ib_dim) // 2
    # sets the parameters of IB or MLP baseline.
    config.ib = args.ib
    config.activation = args.activation
    config.hidden_dim = args.hidden_dim
    config.ib_dim = args.ib_dim
    config.beta = args.beta
    config.sample_size = args.sample_size
    config.kl_annealing = args.kl_annealing
    config.deterministic = args.deterministic
    if args.dropout is not None:
        config.hidden_dropout_prob = args.dropout

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

    if args.mixout > 0:
        from mixout import MixLinear
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear) and not ('output' in name and 'attention' not in name):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout
                    ).cuda()
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, eval_type="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.evaluate_after_each_epoch and args.do_train:
       args.do_eval = False 
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, _, _ = evaluate(args, model, tokenizer, prefix=prefix)
            results.update(result)

    return results


if __name__ == "__main__":
    main()
