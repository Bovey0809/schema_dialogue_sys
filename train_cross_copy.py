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

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import pickle as pkl
import socket
import math
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSlotCross
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from nltk.corpus import stopwords

from data_utils_test import Corpus

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
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

print ('Linux host name:', socket.gethostname(), get_host_ip())
print ('use gpu', os.environ['CUDA_VISIBLE_DEVICES'])

class Bert_v1(nn.Module):
	def __init__(self, ninput, nhidden, n_layer=1, dropout=0, device=None):
		super(Bert_v1, self).__init__()
		self.nhidden = nhidden
		self.ninput = ninput
		self.dropout = nn.Dropout(dropout)
		self.n_layer = n_layer
		self.device = device

		# self.bert = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=None)
		self.bert = BertForSlotCross.from_pretrained(args.load_model_dir, \
												cache_dir=args.load_model_dir)

	def forward(self, input_ids, segment_ids, input_mask, special_token_ids):

		bsz = input_ids.size(0)
		logits = self.bert(input_ids, segment_ids, input_mask, special_token_ids=special_token_ids)

		assert logits.size(0) == bsz

		return logits

class InputFeatures(object):
	def __init__(self,
				 example_id,
				 seq_features,
				 slot_label,
				 dialogue_id,
				 api_id,
				 cross_api_id,
				 turn_label,
				 slot_in_cross_frame_his,
				 slot_in_frame_his,
				 slot_required,
				 slot_optional,
				 frame_continue,
				 utterance,
				 slot,
				 slot_cross,
				 uttr_tokens
	):
		self.example_id = example_id
		self.seq_features = [
			{
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids,
				'special_tokens_ids':special_tokens_ids,
			}
			for _, input_ids, input_mask, segment_ids, special_tokens_ids in seq_features
		]
		self.slot_label = slot_label
		self.dialogue_id = dialogue_id
		self.api_id = api_id
		self.cross_api_id = cross_api_id
		self.turn_label = turn_label
		self.utterance = utterance
		self.slot = slot
		self.slot_cross = slot_cross
		self.slot_in_cross_frame_his = slot_in_cross_frame_his
		self.slot_in_frame_his = slot_in_frame_his
		self.slot_required = slot_required
		self.slot_optional = slot_optional
		self.frame_continue = frame_continue
		self.uttr_tokens = uttr_tokens

class DataIterator:
	def __init__(self, examples, tokenizer, max_numVals_of_slot, max_seq_length,
				is_training, max_uttr_len=64, eval=False, batch_size=16, shuffle=False):
		self.ori_examples = examples
		self.tokenizer = tokenizer
		self.max_numVals_of_slot = max_numVals_of_slot
		self.max_seq_length = max_seq_length
		self.is_training = is_training
		self.max_uttr_len = max_uttr_len
		self.eval = eval
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.show_example = True

		self.idx = 0
		self.end_of_data = False
		self.data_buffer = []
		self.k = batch_size*100

		self.filt_data()
		self.num_examples_valid = len(self.examples)

	def filt_data(self):
		self.examples = copy.deepcopy(self.ori_examples)
		for i in list(range(len(self.examples)))[::-1]:
			# if i%5000==0:
			# 	print(i)
			example = self.examples[i]
			if example[-1] != 0 and not example[-2] and not self.eval:
				p = random.uniform(0,1)
				if p > 0.01:
					# self.examples.pop(i)
					del self.examples[i]

	def __iter__(self):
		return self

	def __len__(self):
		return self.num_examples_valid

	def reset(self):
		self.idx = 0

	def __next__(self):
		if self.end_of_data:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		self.batch = []

		if len(self.data_buffer) == 0:
			self.data_buffer = copy.deepcopy(self.examples[self.idx:self.idx+self.k])
			self.idx += self.k
	
			if self.shuffle:
				random.shuffle(self.data_buffer)

		if len(self.data_buffer) == 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		try:
			while True:
				try:
					example = self.data_buffer.pop(0)
				except IndexError:
					break

				self.batch.append(self.convert_example_to_feature(example))

				if len(self.batch) >= self.batch_size:
					break

		except IndexError:
			self.end_of_data = True

		if len(self.batch) == 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		return self.batchify(self.batch)


	def batchify(self, data):
		all_input_ids = select_field(data, 'input_ids') 
		all_input_mask = select_field(data, 'input_mask')
		all_segment_ids = select_field(data, 'segment_ids')
		all_special_tokens_ids = select_field(data, 'special_tokens_ids')

		all_slot_label = [f.slot_label for f in data] # bsz

		all_dialogue_id = [f.dialogue_id for f in data]
		all_api_id = [f.api_id for f in data]
		all_cross_api_id = [f.cross_api_id for f in data]
		all_turn_label = [f.turn_label for f in data]
		all_utterance = [f.utterance for f in data]
		all_slot = [f.slot for f in data]
		all_slot_cross = [f.slot_cross for f in data]
		all_slot_in_cross_frame_his = [f.slot_in_cross_frame_his for f in data]
		all_slot_in_frame_his = [f.slot_in_frame_his for f in data]
		all_slot_required = [f.slot_required for f in data]
		all_slot_optional = [f.slot_optional for f in data]
		all_frame_continue = [f.frame_continue for f in data]
		all_uttr_tokens = [f.uttr_tokens for f in data]

		input_ids = torch.tensor(all_input_ids, dtype=torch.long)
		input_mask = torch.tensor(all_input_mask, dtype=torch.float)
		segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
		special_tokens_ids = torch.tensor(all_special_tokens_ids, dtype=torch.long)

		input_ids = input_ids.squeeze(1)
		segment_ids = segment_ids.squeeze(1)
		input_mask = input_mask.squeeze(1)
		special_tokens_ids = special_tokens_ids.squeeze(1)

		slot_label = torch.tensor(all_slot_label, dtype=torch.long)

		dialogue_id = all_dialogue_id
		api_id = all_api_id
		cross_api_id = all_cross_api_id
		turn_label = all_turn_label

		utterance = all_utterance
		slot = all_slot
		slot_cross = all_slot_cross
		
		slot_in_cross_frame_his = all_slot_in_cross_frame_his
		slot_in_frame_his = all_slot_in_frame_his
		slot_required = all_slot_required
		slot_optional = all_slot_optional
		frame_continue = all_frame_continue

		uttr_tokens = all_uttr_tokens

		return [input_ids, input_mask, segment_ids, special_tokens_ids, slot_label, \
			   dialogue_id, api_id, cross_api_id, turn_label, utterance, uttr_tokens, \
			   slot, slot_cross, slot_in_cross_frame_his, \
			   slot_in_frame_his, slot_required, slot_optional, frame_continue]


	def convert_example_to_feature(self, example):
		seq_features = []
		dialogue_id = example[0]
		api_id = example[1]
		cross_api_id = example[2]
		turn_label = example[3]
		utterance = example[4]
		slot = example[5]
		slot_cross = example[6]
		slot_in_cross_frame_his = example[7]
		slot_in_frame_his = example[8]
		slot_required = example[9]
		slot_optional = example[10]
		frame_continue = example[11]

		uttr_tokens = example[12][-self.max_uttr_len:]
		slot_tokens = example[13]
		slot_cross_tokens = example[14]

		special_tokens = []
		special_tokens_ids = 0
		if slot_in_cross_frame_his:
			special_tokens_ids += 1
		if slot_in_frame_his:
			special_tokens_ids += 2
		if slot_required:
			special_tokens_ids += 4
		if slot_optional:
			special_tokens_ids += 8
		if frame_continue:
			special_tokens_ids += 16

		# Modifies `context_tokens_choice` and `ending_tokens` in
		# _truncate_seq_pair(pre_tokens, hyp_tokens, max_seq_length - 3)
		uttr_tokens_padded = uttr_tokens + ['[PAD]']*(self.max_uttr_len-len(uttr_tokens))
		
		tokens = ['[CLS]'] + uttr_tokens_padded + ['[SEP]'] + slot_tokens \
								+ ['[SEP]'] + slot_cross_tokens + ['[SEP]']

		# segment_ids = [0] * (len(uttr_tokens_padded) + 2) + [1] * (len(slot_tokens) + len(special_tokens) + 2)
		segment_ids = [0] * (len(uttr_tokens_padded) + len(slot_tokens) + 3) + [1] * (len(slot_cross_tokens) + 1)
		special_tokens_ids = [special_tokens_ids] * self.max_seq_length

		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

			
		# manner1
		# uttr_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - len(uttr_tokens)) + \
		# 			[1] * (3 + len(slot_tokens))
		# uttr_att_mask += [0] * (self.max_seq_length - len(uttr_att_mask))

		# manner2
		uttr_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - len(uttr_tokens)) + \
					[1] * (2 + len(slot_tokens) + len(slot_cross_tokens))
		uttr_att_mask += [0] * (self.max_seq_length - len(uttr_att_mask))

		slot_att_mask = uttr_att_mask
		cls_att_mask = uttr_att_mask
	
		sep1_att_mask = uttr_att_mask
		sep2_att_mask = uttr_att_mask
		sep3_att_mask = uttr_att_mask

		input_mask = []
		input_mask.append(cls_att_mask)
		input_mask += [uttr_att_mask] * self.max_uttr_len
		input_mask.append(sep1_att_mask)
		input_mask += [slot_att_mask] * (len(slot_tokens))
		input_mask.append(sep2_att_mask)
		input_mask += [slot_att_mask] * (len(slot_cross_tokens))
		input_mask.append(sep3_att_mask)

		input_mask += [[0]*self.max_seq_length] * (self.max_seq_length - len(input_mask))

		# Zero-pad up to the sequence length.
		padding = [0] * (self.max_seq_length - len(input_ids))
		input_ids += padding
		segment_ids += padding

		# if len(input_ids) > 128:
		# 	print(tokens)
		assert len(input_ids) == self.max_seq_length
		assert len(input_mask) == self.max_seq_length
		assert len(segment_ids) == self.max_seq_length

		seq_features.append((tokens, input_ids, input_mask, \
							 segment_ids, special_tokens_ids))

		if self.is_training:
			slot_label = example[-1]

		if self.show_example:
			self.show_example = False
			logger.info("*** Example ***")
			logger.info(f"dialog_id: {example[0]}")
			for (tokens, input_ids, input_mask, segment_ids, special_tokens_ids) in seq_features:
				logger.info(f"tokens: {' '.join(tokens)}")
				logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
				# logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
				logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
			if self.is_training:
				logger.info(f"slot_label: {slot_label}")

		return InputFeatures(
				example_id = example[0],
				seq_features = seq_features,
				slot_label = slot_label,
				dialogue_id = dialogue_id,
				api_id = api_id,
				cross_api_id = cross_api_id,
				turn_label = turn_label,
				utterance = utterance,
				slot = slot,
				slot_cross = slot_cross,
				slot_in_cross_frame_his = slot_in_cross_frame_his,
				slot_in_frame_his = slot_in_frame_his,
				slot_required = slot_required,
				slot_optional = slot_optional,
				frame_continue = frame_continue,
				uttr_tokens = uttr_tokens,
			)

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

def select_field(features, field):
	return [
		[
			choice[field]
			for choice in feature.seq_features
		]
		for feature in features
	]

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x

def calc_loss(logits, slot_label):
	loss = F.cross_entropy(logits, slot_label)

	return loss

def accuracy(logits, slot_label):
	temp_accuracy = np.zeros(2)
	temp_examples = np.zeros(2)
	_, preds = logits.max(1)
	probs = F.softmax(logits, 1)
	for i in range(2):
		temp_accuracy[i] = ((preds == slot_label) * (slot_label == i)) \
								.float().cpu().numpy().sum()
		temp_examples[i] = (slot_label == i).float().cpu().numpy().sum()
		
	preds = list(preds.cpu().numpy())
	ground = list(slot_label.cpu().numpy())

	score = list(probs[:,0].detach().cpu().numpy())
	
	return temp_accuracy, temp_examples, preds, ground, score


def train(train_features, epoch, global_step):

	logger.info("\ntraining epoch %d, Num examples = %d, Batch size = %d, Num steps = %d" \
		%(epoch, len(train_features), args.train_batch_size, num_train_steps))

	model.train()
	# for _ in trange(int(args.num_train_epochs), desc="Epoch"):
	tr_loss = 0
	tr_accuracy = np.zeros(2)
	nb_examples = np.zeros(2)

	nb_tr_steps = 0

	print_steps = 200
	tmp_accuracy = np.zeros(2)
	tmp_examples = np.zeros(2)

	tmp_loss = 0

	for step, batch in enumerate(train_features):
		batch[:5] = tuple(t.to(device) for t in batch[:5])
		input_ids, input_mask, segment_ids, special_token_ids, slot_label = batch[:5]
		bsz = input_ids.size(0)

		logits = model(input_ids, segment_ids, input_mask, special_token_ids)

		loss = calc_loss(logits, slot_label)

		temp_accuracy, temp_examples, _, _, _ = accuracy(logits, slot_label)


		if n_gpu > 1:
			loss = loss.mean() # mean() to average on multi-gpu.
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

		if nb_tr_steps%print_steps == 0:
			tmp_accuracy = tmp_accuracy/tmp_examples
			tmp_loss /= print_steps

			logger.info('steps:%d/total_steps:%d, loss:%.3f, acc:%s'% \
								(global_step, num_train_steps, tmp_loss, str(tmp_accuracy)))
									
			tmp_accuracy = np.zeros(2)
			tmp_examples = np.zeros(2)
			tmp_loss = 0

		if args.fp16:
			optimizer.backward(loss)
		else:
			loss.backward()
		if (step + 1) % args.gradient_accumulation_steps == 0:
			# modify learning rate with special warm up BERT uses
			lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
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
	tr_accuracy = np.zeros(2)
	nb_examples = np.zeros(2)

	nb_tr_steps = 0

	total = math.ceil(len(eval_features)/args.train_batch_size)
	n = 0

	total_preds = []

	total_dialogue_id = []
	total_api_id = []
	total_cross_api_id = []
	total_turn_label = []
	total_utterance = []
	total_uttr_tokens = []
	total_slots = []
	total_slots_cross = []
	total_slot_in_corss_frame_his = []
	total_slot_in_frame_his = []
	total_slot_required = []
	total_slot_optional = []
	total_frame_continue = []

	total_ground = []
	total_score = []

	eval_loss = 0

	for sample_idx, batch in enumerate(eval_features):
		batch[:5] = tuple(t.to(device) for t in batch[:5])
		input_ids, input_mask, segment_ids, special_token_ids, slot_label, \
		dialogue_id, api_id, cross_api_id, turn_label, utterance, uttr_tokens, slot, slot_cross, \
		slot_in_cross_frame_his, slot_in_frame_his, slot_required, slot_optional, frame_continue = batch

		bsz = input_ids.size(0)
		
		with torch.no_grad():
			logits = model(input_ids, segment_ids, input_mask, special_token_ids)

			loss = calc_loss(logits, slot_label)

			temp_accuracy, temp_examples, preds, ground, score = accuracy(logits, slot_label)

		eval_loss += loss.mean().item()
		nb_examples += temp_examples

		nb_tr_steps += 1

		tr_accuracy += temp_accuracy

		total_dialogue_id += dialogue_id
		total_api_id += api_id
		total_cross_api_id += cross_api_id
		total_turn_label += turn_label
		total_utterance += utterance
		total_uttr_tokens += uttr_tokens
		total_slots += slot
		total_slots_cross += slot_cross
		total_slot_in_corss_frame_his += slot_in_cross_frame_his
		total_slot_in_frame_his += slot_in_frame_his
		total_slot_required += slot_required
		total_slot_optional += slot_optional
		total_frame_continue += frame_continue

		total_ground += ground
		total_preds += preds
		total_score += score

		# break

	tr_accuracy /= nb_examples

	eval_loss /= nb_tr_steps

	total_results = (total_dialogue_id, total_api_id, total_cross_api_id, \
					total_turn_label, total_utterance, total_uttr_tokens, \
					total_slots, total_slots_cross, \
					total_slot_in_corss_frame_his, total_slot_in_frame_his, \
					total_slot_required, total_slot_optional, total_frame_continue, \
					total_score, total_ground, total_preds)


	return eval_loss, tr_accuracy, nb_examples, total_results


def arg_parser():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .csv files (or other data files) for the task.")

	parser.add_argument("--load_model_dir",
						default=None,
						type=str,
						help="Where to load and cache the model file.")

	parser.add_argument("--history_model_file",
						default=None,
						type=str,
						help="Where to load the history model.")

	parser.add_argument("--bert_model", default=None, type=str, required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
							 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model checkpoints will be written.")

	## Other parameters
	parser.add_argument("--max_seq_length",
						default=64,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
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
	parser.add_argument("--do_lower_case",
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
	parser.add_argument("--warmup_proportion",
						default=0.5,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
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
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						default=False,
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	parser.add_argument("--cross_folds",
						default=10,
						type=int,
						help="")

	args = parser.parse_args()

	return args

#-----------------------------main-------------------------------------------
args = arg_parser()

if args.local_rank == -1 or args.no_cuda:
	device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	n_gpu = torch.cuda.device_count()
else:
	torch.cuda.set_device(args.local_rank)
	device = torch.device("cuda", args.local_rank)
	n_gpu = 1
	# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
	torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
	device, n_gpu, bool(args.local_rank != -1), args.fp16))

if args.gradient_accumulation_steps < 1:
	raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
						args.gradient_accumulation_steps))

args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

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
corpus = Corpus(args, max_uttr_len=96)
# pkl.dump(corpus, open('corpus.pkl', 'wb'))
# exit()
# corpus = pkl.load(open('corpus_sample0p99Dev.pkl', 'rb'))
corpus.max_uttr_len = 96
train_data, dev_data, test_data = corpus.get_slotCross_set()

print('#train:', len(train_data))
print('#dev:', len(dev_data))
print('#test:', len(test_data))
corpus.check_length(train_data, maxlen = corpus.max_uttr_len, idx=-5)

# exit()
train_features = DataIterator(train_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=False, batch_size=args.train_batch_size, shuffle=True)
dev_features = DataIterator(dev_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=True, batch_size=args.train_batch_size, shuffle=False)
test_features = DataIterator(test_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=True, batch_size=args.train_batch_size, shuffle=False)
print('#filtered train:', len(train_features))
print('#filtered dev:', len(dev_features))
print('#filtered test:', len(test_features))

num_train_steps = int(len(train_features) / args.train_batch_size / \
						args.gradient_accumulation_steps * args.num_train_epochs)

#-------------------------------------------------------------------------------
model = Bert_v1(768, 300, n_layer=1, dropout=0.1, device=device)
print(model)
if args.fp16:
	model.half()


if args.history_model_file is not None:
	model_state_dict = torch.load(args.history_model_file)
	model.load_state_dict(model_state_dict)
	print('#Finished loading history model file from %s !'%args.history_model_file)


model.to(device)
if args.local_rank != -1:
	try:
		from apex.parallel import DistributedDataParallel as DDP
	except ImportError:
		raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

	model = DDP(model)
elif n_gpu > 1:
	model = torch.nn.DataParallel(model)

param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

t_total = num_train_steps
global_step = 0

if args.local_rank != -1:
	t_total = t_total // torch.distributed.get_world_size()
if args.fp16:
	try:
		from apex.optimizers import FP16_Optimizer
		from apex.optimizers import FusedAdam
	except ImportError:
		raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

	optimizer = FusedAdam(optimizer_grouped_parameters,
						  lr=args.learning_rate,
						  bias_correction=False,
						  max_grad_norm=1.0)
	if args.loss_scale == 0:
		optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
	else:
		optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
else:
	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr=args.learning_rate,
						 warmup=args.warmup_proportion,
						 t_total=t_total)
for epoch in range(int(args.num_train_epochs)):

	if args.do_train:
		train_features.filt_data()
		# random.shuffle(train_examples)
		tr_loss, tr_accuracy, global_step, \
			tr_examples = train(train_features, epoch, global_step)
	else:
		tr_examples, tr_loss, tr_accuracy,  = 0, 999., 0.
	if epoch > -1:
		eval_loss, eval_accuracy, eval_examples, total_eval_results = evaluate(dev_features)

		# test_loss, test_accuracy, test_examples, total_test_results = evaluate(test_features)


		logging.info('\n'+'#'*40)
		logging.info('epoch%d, train_examples:%s, train_acc:%s' \
				%(epoch, str(tr_examples), str(tr_accuracy)))
		logging.info('epoch%d, dev_examples:%s, dev_acc:%s' \
				%(epoch, str(eval_examples), str(eval_accuracy)))
		# logging.info('epoch%d, test_examples:%s, test_acc:%s' \
		# 		%(epoch, str(test_examples), str(test_accuracy)))

		logging.info('#'*40)


now = datetime.now()
strnow = datetime.strftime(now,'%Y-%m-%d_%H_%M_%S_')
eval_results_path = 'detailed_results/'+strnow+'total_eval_results.pkl'
test_results_path = 'detailed_results/'+strnow+'total_test_results.pkl'

pkl.dump(total_eval_results, open(eval_results_path, 'wb'))
# pkl.dump(total_test_results, open(test_results_path, 'wb'))
print('Save eval results to %s'%eval_results_path)
# print('Save test results to %s'%test_results_path)

if args.do_train:
	now = datetime.now()
	strnow = datetime.strftime(now,'%Y-%m-%d_%H_%M_%S_')
	model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
	output_model_file = os.path.join(args.output_dir, strnow+"pytorch_model.bin")
	print('Saved model file:%s', output_model_file)
	if args.do_train:
	    torch.save(model_to_save.state_dict(), output_model_file)


