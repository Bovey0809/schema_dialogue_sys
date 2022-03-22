# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import argparse
import copy
import csv
import logging
import math
import os

import pickle as pkl
import random
import socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nltk.corpus import stopwords
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from data_utils_test import Corpus
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForIntent
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

avg_len = []


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


print('Linux host name:', socket.gethostname(), get_host_ip())
# TODO CUDA devices can't be printed.
# print('use gpu', os.environ['CUDA_VISIBLE_DEVICES'])


class Bert_v1(nn.Module):
    def __init__(self, ninput, nhidden, n_layer=1, dropout=0, device=None):
        super(Bert_v1, self).__init__()
        self.nhidden = nhidden
        self.ninput = ninput
        self.dropout = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.device = device

        # self.bert = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=None)
        self.bert = BertForIntent.from_pretrained(
            args.bert_model, cache_dir=args.load_model_dir)

    def forward(self, input_ids, segment_ids, input_mask, intent_mask,
                special_token_ids):

        bsz = input_ids.size(0)
        logits = self.bert(input_ids,
                           segment_ids,
                           input_mask,
                           intent_mask,
                           special_token_ids=special_token_ids)

        assert logits.size(0) == bsz

        return logits


class InputFeatures(object):
    def __init__(
        self,
        example_id,
        seq_features,
        dialogue_id,
        api_id,
        turn_id,
        utterance,
        uttr_tokens,
        intents,
        intent_label,
        last_intent_label,
    ):
        self.example_id = example_id
        self.seq_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'intent_mask': intent_mask,
            'special_token_ids': special_token_ids,
        } for _, input_ids, input_mask, segment_ids, intent_mask,
                             special_token_ids in seq_features]
        self.dialogue_id = dialogue_id
        self.api_id = api_id,
        self.turn_id = turn_id
        self.utterance = utterance
        self.uttr_tokens = uttr_tokens
        self.intents = intents
        self.intent_label = intent_label
        self.last_intent_label = last_intent_label


def convert_examples_to_features(examples, tokenizer, max_numIntents, max_seq_length, \
 is_training, max_uttr_len=64, eval=False):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in enumerate(examples):
        seq_features = []

        dialogue_id = example[0]
        api_id = example[1]
        turn_id = example[2]
        utterance = example[3]

        intents = example[5]

        uttr_tokens = example[4][-max_uttr_len:]

        example[6] += [['[PAD]']] * (max_numIntents - len(example[6]))
        example[8] += [0] * (max_numIntents - len(example[8]))

        assert len(example[6]) == max_numIntents
        intent_desp = copy.deepcopy(example[6])

        for i in range(1, len(intent_desp))[::-1]:
            intent_desp.insert(i, ['[SEP]'])

        for val in intent_desp[1:]:
            intent_desp[0] += val
        intent_desp = intent_desp[0]

        # Modifies `context_tokens_choice` and `ending_tokens` in
        # _truncate_seq_pair(pre_tokens, hyp_tokens, max_seq_length - 3)
        uttr_tokens_padded = uttr_tokens + ['[PAD]'] * (max_uttr_len -
                                                        len(uttr_tokens))
        tokens = ['[CLS]'] + uttr_tokens_padded + ['[SEP]'
                                                   ] + intent_desp + ['[SEP]']

        segment_ids = [0] * (len(uttr_tokens_padded) +
                             2) + [1] * (len(intent_desp) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        intent_mask = []
        intent_lens = [len(val) for val in example[6]]
        for i in range(len(example[6])):
            # mask_style_1
            tmp_intent_mask = [0] * (len(uttr_tokens_padded) + 2 + \
             sum(intent_lens[:i]) + i) + [1] * intent_lens[i]
            tmp_intent_mask += [0] * (max_seq_length - len(tmp_intent_mask))
            intent_mask.append(tmp_intent_mask)

        special_token_ids = [0] * (len(uttr_tokens_padded) + 2)
        for i in range(len(example[6])):
            special_token_ids += [example[8][i]] * len(example[6][i]) + [0]
        special_token_ids += [0] * (max_seq_length - len(special_token_ids))

        # manner1
        uttr_att_mask = [1] * (1 + len(uttr_tokens)) + [0] * (max_uttr_len - len(uttr_tokens)) + \
           [1] * (2 + len(intent_desp))
        uttr_att_mask += [0] * (max_seq_length - len(uttr_att_mask))

        # manner2
        # uttr_att_mask = [0] + [1] * (len(uttr_tokens)) + [0] * (max_uttr_len - len(uttr_tokens)) + \
        # 			[1] * (2 + len(slot_tokens))
        # uttr_att_mask += [0] * (max_seq_length - len(uttr_att_mask))

        slot_att_mask = uttr_att_mask
        cls_att_mask = uttr_att_mask

        sep1_att_mask = uttr_att_mask
        sep2_att_mask = uttr_att_mask

        input_mask = []
        input_mask.append(cls_att_mask)
        input_mask += [uttr_att_mask] * max_uttr_len
        input_mask.append(sep1_att_mask)

        for i in range(len(example[6])):
            # mask_style_1
            tmp_val_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (max_uttr_len - \
                  len(uttr_tokens))
            tmp_val_att_mask += [0] * (sum(intent_lens[:i]) + i)
            tmp_val_att_mask += [1] * intent_lens[i]
            tmp_val_att_mask += [0] * (max_seq_length - len(tmp_val_att_mask))

            input_mask += [tmp_val_att_mask] * intent_lens[i]
            if i < len(example[6]) - 1:
                input_mask.append(cls_att_mask)

        input_mask.append(sep2_att_mask)
        input_mask += [[0] * max_seq_length
                       ] * (max_seq_length - len(input_mask))

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        segment_ids += padding

        # if len(input_ids) > 128:
        # 	print(tokens)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        seq_features.append((tokens, input_ids, input_mask, segment_ids,
                             intent_mask, special_token_ids))

        if is_training:
            intent_label = example[7]

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"dialog_id: {example[0]}")
            for (tokens, input_ids, input_mask, segment_ids, intent_mask,
                 special_token_ids) in seq_features:
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                # logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"intent_label: {intent_label}")

        features.append(
            InputFeatures(example_id=example[0],
                          seq_features=seq_features,
                          dialogue_id=dialogue_id,
                          api_id=api_id,
                          turn_id=turn_id,
                          utterance=utterance,
                          uttr_tokens=uttr_tokens,
                          intents=intents,
                          intent_label=intent_label,
                          last_intent_label=-1))

    return features

def convert_examples_to_features_infer(examples, tokenizer, max_numIntents, max_seq_length, \
 is_training, max_uttr_len=64, eval=False):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in enumerate(examples):

        dialogue_id = example[0]
        api_id = example[1]
        turn_id = example[2]
        utterance = example[3]

        intents = example[5]

        uttr_tokens = example[4][-max_uttr_len:]

        example[6] += [['[PAD]']] * (max_numIntents - len(example[6]))
        example[8] += [0] * (max_numIntents - len(example[8]))

        for last_intent_label in range(len(example[6])):
            seq_features = []
            assert len(example[6]) == max_numIntents
            intent_desp = copy.deepcopy(example[6])

            for i in range(1, len(intent_desp))[::-1]:
                intent_desp.insert(i, ['[SEP]'])

            for val in intent_desp[1:]:
                intent_desp[0] += val
            intent_desp = intent_desp[0]

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # _truncate_seq_pair(pre_tokens, hyp_tokens, max_seq_length - 3)
            uttr_tokens_padded = uttr_tokens + ['[PAD]'] * (max_uttr_len -
                                                            len(uttr_tokens))
            tokens = ['[CLS]'] + uttr_tokens_padded + [
                '[SEP]'
            ] + intent_desp + ['[SEP]']

            segment_ids = [0] * (len(uttr_tokens_padded) +
                                 2) + [1] * (len(intent_desp) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            intent_mask = []
            intent_lens = [len(val) for val in example[6]]
            for i in range(len(example[6])):
                # mask_style_1
                tmp_intent_mask = [0] * (len(uttr_tokens_padded) + 2 + \
                 sum(intent_lens[:i]) + i) + [1] * intent_lens[i]
                tmp_intent_mask += [0
                                    ] * (max_seq_length - len(tmp_intent_mask))
                intent_mask.append(tmp_intent_mask)

            special_token_ids = [0] * (len(uttr_tokens_padded) + 2)
            for i in range(len(example[6])):
                special_token_ids += [int(i == last_intent_label)] * len(
                    example[6][i]) + [0]
            special_token_ids += [0
                                  ] * (max_seq_length - len(special_token_ids))

            # manner1
            uttr_att_mask = [1] * (1 + len(uttr_tokens)) + [0] * (max_uttr_len - len(uttr_tokens)) + \
               [1] * (2 + len(intent_desp))
            uttr_att_mask += [0] * (max_seq_length - len(uttr_att_mask))

            # manner2
            # uttr_att_mask = [0] + [1] * (len(uttr_tokens)) + [0] * (max_uttr_len - len(uttr_tokens)) + \
            # 			[1] * (2 + len(slot_tokens))
            # uttr_att_mask += [0] * (max_seq_length - len(uttr_att_mask))

            slot_att_mask = uttr_att_mask
            cls_att_mask = uttr_att_mask

            sep1_att_mask = uttr_att_mask
            sep2_att_mask = uttr_att_mask

            input_mask = []
            input_mask.append(cls_att_mask)
            input_mask += [uttr_att_mask] * max_uttr_len
            input_mask.append(sep1_att_mask)

            for i in range(len(example[6])):
                # mask_style_1
                tmp_val_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (max_uttr_len - \
                      len(uttr_tokens))
                tmp_val_att_mask += [0] * (sum(intent_lens[:i]) + i)
                tmp_val_att_mask += [1] * intent_lens[i]
                tmp_val_att_mask += [0] * (max_seq_length -
                                           len(tmp_val_att_mask))

                input_mask += [tmp_val_att_mask] * intent_lens[i]
                if i < len(example[6]) - 1:
                    input_mask.append(cls_att_mask)

            input_mask.append(sep2_att_mask)
            input_mask += [[0] * max_seq_length
                           ] * (max_seq_length - len(input_mask))

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            segment_ids += padding

            # if len(input_ids) > 128:
            # 	print(tokens)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            seq_features.append((tokens, input_ids, input_mask, segment_ids,
                                 intent_mask, special_token_ids))

            if is_training:
                intent_label = example[7]

            if example_index < 1:
                logger.info("*** Example ***")
                logger.info(f"dialog_id: {example[0]}")
                for (tokens, input_ids, input_mask, segment_ids, intent_mask,
                     special_token_ids) in seq_features:
                    logger.info(f"tokens: {' '.join(tokens)}")
                    logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                    # logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                    logger.info(
                        f"segment_ids: {' '.join(map(str, segment_ids))}")
                if is_training:
                    logger.info(f"intent_label: {intent_label}")

            features.append(
                InputFeatures(example_id=example[0],
                              seq_features=seq_features,
                              dialogue_id=dialogue_id,
                              api_id=api_id,
                              turn_id=turn_id,
                              utterance=utterance,
                              uttr_tokens=uttr_tokens,
                              intents=intents,
                              intent_label=intent_label,
                              last_intent_label=last_intent_label))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def batchify(data, batch_size=16, shuffle=True):
    all_input_ids = select_field(data, 'input_ids')
    all_input_mask = select_field(data, 'input_mask')
    all_segment_ids = select_field(data, 'segment_ids')
    all_intent_mask = select_field(data, 'intent_mask')
    all_special_token_ids = select_field(data, 'special_token_ids')

    all_intent_label = [f.intent_label for f in data]  # bsz

    all_dialogue_id = [f.dialogue_id for f in data]
    all_api_id = [f.api_id for f in data]
    all_turn_id = [f.turn_id for f in data]
    all_utterance = [f.utterance for f in data]
    all_uttr_tokens = [f.uttr_tokens for f in data]
    all_intents = [f.intents for f in data]
    all_last_intent_label = [f.last_intent_label for f in data]  # bsz

    if shuffle:
        idx = list(range(len(all_turn_id)))
        random.shuffle(idx)
        all_input_ids = [all_input_ids[i] for i in idx]
        all_input_mask = [all_input_mask[i] for i in idx]
        all_segment_ids = [all_segment_ids[i] for i in idx]
        all_intent_mask = [all_intent_mask[i] for i in idx]
        all_special_token_ids = [all_special_token_ids[i] for i in idx]

        all_intent_label = [all_intent_label[i] for i in idx]

        all_dialogue_id = [all_dialogue_id[i] for i in idx]
        all_api_id = [all_api_id[i] for i in idx]
        all_turn_id = [all_turn_id[i] for i in idx]

        all_utterance = [all_utterance[i] for i in idx]
        all_uttr_tokens = [all_uttr_tokens[i] for i in idx]
        all_intents = [all_intents[i] for i in idx]
        all_last_intent_label = [all_last_intent_label[i] for i in idx]

    for i in range(math.ceil(1. * len(all_input_ids) / batch_size)):
        input_ids = torch.tensor(all_input_ids[i * batch_size:(i + 1) *
                                               batch_size],
                                 dtype=torch.long)
        input_mask = torch.tensor(all_input_mask[i * batch_size:(i + 1) *
                                                 batch_size],
                                  dtype=torch.float)
        segment_ids = torch.tensor(all_segment_ids[i * batch_size:(i + 1) *
                                                   batch_size],
                                   dtype=torch.long)
        intent_mask = torch.tensor(all_intent_mask[i * batch_size:(i + 1) *
                                                   batch_size],
                                   dtype=torch.float)
        special_token_ids = torch.tensor(
            all_special_token_ids[i * batch_size:(i + 1) * batch_size],
            dtype=torch.long)

        input_ids = input_ids.squeeze(1)
        segment_ids = segment_ids.squeeze(1)
        input_mask = input_mask.squeeze(1)
        intent_mask = intent_mask.squeeze(1)
        special_token_ids = special_token_ids.squeeze(1)

        intent_label = torch.tensor(all_intent_label[i * batch_size:(i + 1) *
                                                     batch_size],
                                    dtype=torch.long)

        dialogue_id = all_dialogue_id[i * batch_size:(i + 1) * batch_size]
        api_id = all_api_id[i * batch_size:(i + 1) * batch_size]
        turn_id = all_turn_id[i * batch_size:(i + 1) * batch_size]

        utterance = all_utterance[i * batch_size:(i + 1) * batch_size]
        uttr_tokens = all_uttr_tokens[i * batch_size:(i + 1) * batch_size]
        intents = all_intents[i * batch_size:(i + 1) * batch_size]
        last_intent_label = all_last_intent_label[i * batch_size:(i + 1) *
                                                  batch_size]

        yield [input_ids, input_mask, segment_ids, intent_mask, special_token_ids, intent_label, \
           dialogue_id, api_id, turn_id, utterance, uttr_tokens, intents, last_intent_label]


def select_field(features, field):
    return [[choice[field] for choice in feature.seq_features]
            for feature in features]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def calc_loss(logits, intent_label):
    loss = F.cross_entropy(logits, intent_label)

    return loss


def accuracy(logits, intent_label):
    temp_accuracy = 0
    temp_examples = 0
    _, preds = logits.max(1)

    temp_accuracy = (preds == intent_label).float().cpu().numpy().sum()
    temp_examples = preds.size(0)

    preds = list(preds.cpu().numpy())
    ground = list(intent_label.cpu().numpy())

    return temp_accuracy, temp_examples, preds, ground


def train(train_features, epoch, global_step):

    logger.info("\ntraining epoch %d, Num examples = %d, Batch size = %d, Num steps = %d" \
     %(epoch, len(train_features), args.train_batch_size, num_train_steps))

    model.train()
    # for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    tr_loss = 0
    tr_accuracy = 0
    nb_examples = 0

    nb_tr_steps = 0

    print_steps = 200
    tmp_accuracy = 0
    tmp_examples = 0

    tmp_loss = 0

    for step, batch in enumerate(
            batchify(train_features, batch_size=args.train_batch_size)):
        batch[:6] = tuple(t.to(device) for t in batch[:6])
        input_ids, input_mask, segment_ids, intent_mask, special_token_ids, intent_label = batch[:
                                                                                                 6]
        bsz = input_ids.size(0)

        logits = model(input_ids, segment_ids, input_mask, intent_mask,
                       special_token_ids)

        loss = calc_loss(logits, intent_label)

        temp_accuracy, temp_examples, _, _ = accuracy(logits, intent_label)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.fp16 and args.loss_scale != 1.0:
            # rescale loss for fp16 training
            # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
            loss = loss * args.loss_scale
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()

        nb_examples += temp_examples
        nb_tr_steps += 1

        tr_accuracy += temp_accuracy

        tmp_examples += temp_examples
        tmp_accuracy += temp_accuracy

        tmp_loss += loss.item()

        if nb_tr_steps % print_steps == 0:
            tmp_accuracy = tmp_accuracy / tmp_examples
            tmp_loss /= print_steps

            logger.info('steps:%d/total_steps:%d, loss:%.3f, acc:%s'%(global_step, num_train_steps, \
               tmp_loss, str(tmp_accuracy)))

            tmp_accuracy = 0
            tmp_examples = 0
            tmp_loss = 0

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as loss:
                loss.backward()
        else:
            loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(
                global_step / t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # break

    tr_accuracy /= nb_examples
    tr_loss /= nb_tr_steps

    return tr_loss, tr_accuracy, global_step, nb_examples


def evaluate(eval_features):

    model.eval()
    tr_loss = 0
    tr_accuracy = 0
    nb_examples = 0

    nb_tr_steps = 0

    total = math.ceil(len(eval_features) / args.train_batch_size)
    n = 0

    total_preds = []

    total_dialogue_id = []
    total_api_id = []
    total_turn_id = []
    total_utterance = []
    total_uttr_tokens = []
    total_intents = []
    total_last_intent_label = []

    total_ground = []

    eval_loss = 0

    for sample_idx, batch in enumerate(batchify(eval_features, \
         batch_size=args.train_batch_size, shuffle=False)):
        batch[:6] = tuple(t.to(device) for t in batch[:6])
        input_ids, input_mask, segment_ids, intent_mask, special_token_ids, intent_label, \
        dialogue_id, api_id, turn_id, utterance, uttr_tokens, intents, last_intent_label = batch

        bsz = input_ids.size(0)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, intent_mask,
                           special_token_ids)

            loss = calc_loss(logits, intent_label)

            temp_accuracy, temp_examples, preds, ground = accuracy(
                logits, intent_label)

        eval_loss += loss.mean().item()
        nb_examples += temp_examples

        nb_tr_steps += 1

        tr_accuracy += temp_accuracy

        total_dialogue_id += dialogue_id
        total_api_id += api_id
        total_turn_id += turn_id
        total_utterance += utterance
        total_uttr_tokens += uttr_tokens
        total_intents += intents
        total_last_intent_label += last_intent_label

        total_ground += ground
        total_preds += preds

        # break

    tr_accuracy /= nb_examples

    eval_loss /= nb_tr_steps

    total_results = (total_dialogue_id, total_api_id, total_turn_id, total_utterance, total_uttr_tokens, \
        total_intents, total_last_intent_label, total_ground, total_preds)

    return eval_loss, tr_accuracy, nb_examples, total_results


def arg_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the .csv files (or other data files) for the task."
    )

    parser.add_argument("--load_model_dir",
                        default=None,
                        type=str,
                        help="Where to load and cache the model file.")

    parser.add_argument("--history_model_file",
                        default=None,
                        type=str,
                        help="Where to load the history model.")

    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_anoymous",
                        default=False,
                        action='store_true',
                        help=".")
    parser.add_argument("--do_partAnoymous",
                        default=False,
                        action='store_true',
                        help=".")
    parser.add_argument("--do_simple",
                        default=False,
                        action='store_true',
                        help=".")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.5,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        '--fp16',
        default=False,
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--cross_folds", default=10, type=int, help="")

    args, unknown = parser.parse_known_args()
    print(unknown)
    return args


#-----------------------------main-------------------------------------------
args = arg_parser()

if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info(
    "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
    format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
    raise ValueError(
        "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".
        format(args.gradient_accumulation_steps))

args.train_batch_size = int(args.train_batch_size /
                            args.gradient_accumulation_steps)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")
'''
if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
	raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
'''
os.makedirs(args.output_dir, exist_ok=True)

# tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

#---------------- load data ------------------------
# corpus = Corpus(args, max_uttr_len=96)
# pkl.dump(corpus, open('corpus.pkl', 'wb'))
corpus = pkl.load(open('corpus.pkl', 'rb'))
corpus.max_uttr_len = 96
train_data, dev_data, test_data = corpus.get_intent_set()
corpus.check_length(train_data, maxlen=corpus.max_uttr_len)
print('max_numIntents:', corpus.max_numIntents)
print('#train:', len(train_data))
print('#dev:', len(dev_data))
print('#test:', len(test_data))
print("Converting examples to features.(Long Time)")
train_features = convert_examples_to_features(copy.deepcopy(train_data), corpus.tokenizer_bert, \
   corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len)

dev_features = convert_examples_to_features_infer(copy.deepcopy(dev_data), corpus.tokenizer_bert, \
   corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len, eval=True)

test_features = convert_examples_to_features_infer(copy.deepcopy(test_data), corpus.tokenizer_bert, \
   corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len, eval=True)


num_train_steps = int(len(train_features) / args.train_batch_size / \
      args.gradient_accumulation_steps * args.num_train_epochs)

#-------------------------------------------------------------------------------
model = Bert_v1(768, 300, n_layer=1, dropout=0.1, device=device)
print(model)
# if args.fp16:
#     model.half()

if args.history_model_file is not None:
    model_state_dict = torch.load(args.history_model_file)
    model.load_state_dict(model_state_dict)
    print('#Finished loading history model file from %s !' %
          args.history_model_file)

model.to(device)
if args.local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
        )

    model = DDP(model)
# TODO: BUG one gpu training is not possible.
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{
    'params':
    [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay':
    0.01
}, {
    'params':
    [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay':
    0.0
}]

t_total = num_train_steps
global_step = 0

if args.local_rank != -1:
    t_total = t_total // torch.distributed.get_world_size()
if args.fp16:
    try:
        from apex.fp16_utils import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
        )

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False)
    # if args.loss_scale == 0:
    #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    # else:
    #     optimizer = FP16_Optimizer(optimizer,
    #                                static_loss_scale=args.loss_scale)
else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
if args.fp16:
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


for epoch in range(int(args.num_train_epochs)):

    train_features = convert_examples_to_features(copy.deepcopy(train_data), corpus.tokenizer_bert, \
      corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len)

    if args.do_train:
        # random.shuffle(train_examples)
        tr_loss, tr_accuracy, global_step, \
         tr_examples = train(train_features, epoch, global_step)
    else:
        tr_examples, tr_loss, tr_accuracy, = 0, 999., 0.

    eval_loss, eval_accuracy, eval_examples, total_eval_results = evaluate(
        dev_features)

    # test_loss, test_accuracy, test_examples, total_test_results = evaluate(test_features)

    logging.info('\n' + '#' * 40)
    logging.info('epoch%d, train_examples:%d, train_acc:%.4f' \
      %(epoch, tr_examples, tr_accuracy))
    logging.info('epoch%d, dev_examples:%d, dev_acc:%.4f' \
      %(epoch, eval_examples, eval_accuracy))
    # logging.info('epoch%d, test_examples:%d, test_acc:%.4f' \
    # 		%(epoch, test_examples, test_accuracy))

    logging.info('#' * 40)

now = datetime.now()
strnow = datetime.strftime(now, '%Y-%m-%d_%H_%M_%S_')
eval_results_path = 'detailed_results/' + strnow + 'total_eval_results.pkl'
test_results_path = 'detailed_results/' + strnow + 'total_test_results.pkl'

pkl.dump(total_eval_results, open(eval_results_path, 'wb'))
# pkl.dump(total_test_results, open(test_results_path, 'wb'))
print('Save eval results to %s' % eval_results_path)
# print('Save test results to %s'%test_results_path)

if args.do_train:
    now = datetime.now()
    strnow = datetime.strftime(now, '%Y-%m-%d_%H_%M_%S_')
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir,
                                     strnow + "pytorch_model.bin")
    print('Saved model file:%s', output_model_file)
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
