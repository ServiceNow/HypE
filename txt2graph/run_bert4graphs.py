from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.append('..')
sys.path.append('../txt2graph')

from tokenization import BertTokenizer
from data_parallel import DataParallelImbalance
from txt2graph_loader import Txt2GrphDataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="../data/M-FB15K", type=str,
                        help="The input data dir.")
    parser.add_argument("--file", default="train.txt", type=str,
                        help="The input data file name.")
    parser.add_argument("--cache_dir",
                        default='/mnt/home/jonathan/models/txt2graph', type=str,
                        help="The bert cache directory for pretrained model.")
    parser.add_argument("--output_dir",
                        default='/mnt/home/jonathan/models/txt2graph/bert_save', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='/mnt/home/jonathan/models/txt2graph/bert_log', type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--bert_model", default='bert-large-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=1,
                        help="Max tokens that are masked out for prediction.")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # cuda set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir
    )

    train_corpus = os.path.join(args.data_dir, args.file)
    print("Loading Train Dataset", args.data_dir)
    train_dataset = Txt2GrphDataset(
        train_corpus, tokenizer,
        seq_len=args.max_seq_length, corpus_lines=None,
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch


if __name__ == "__main__":
    main()