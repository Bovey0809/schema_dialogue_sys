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
from pytorch_pretrained_bert.modeling import BertForSlotJoint
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
		self.bert = BertForSlotJoint.from_pretrained(args.load_model_dir, \
												cache_dir=args.load_model_dir)


	def forward(self, input_ids, segment_ids, input_mask, uttr_mask, val_mask, special_tokens_ids):

		bsz = input_ids.size(0)
		logits_cls, logits_start, logits_end, logits_cate, logits_req = \
			self.bert(input_ids, segment_ids, input_mask, uttr_mask, val_mask, corpus.max_uttr_len, special_tokens_ids)

		assert logits_cls.size(0) == bsz

		return logits_cls, logits_start, logits_end, logits_cate, logits_req

class InputFeatures(object):
	def __init__(self,
				 example_id,
				 seq_features,
				 slot_label,
				 slot_tag,
				 start,
				 end,
				 cateVal_idx,
				 dialogue_id,
				 api_id,
				 turn_label,
				 slot_in_usr_his,
				 slot_in_sys_his,
				 slot_request_by_sys,
				 utterance,
				 slot,
				 slot_poss_vals,
				 uttr_tokens
	):
		self.example_id = example_id
		self.seq_features = [
			{
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids,
				'uttr_mask': uttr_mask,
				'val_mask': val_mask,
				'special_tokens_ids':special_tokens_ids,
			}
			for _, input_ids, input_mask, segment_ids, uttr_mask, val_mask, special_tokens_ids in seq_features
		]
		self.slot_label = slot_label
		self.slot_tag = slot_tag
		self.start = start
		self.end = end
		self.cateVal_idx = cateVal_idx
		self.dialogue_id = dialogue_id
		self.api_id = api_id
		self.turn_label = turn_label
		self.utterance = utterance
		self.slot = slot
		self.slot_poss_vals = slot_poss_vals
		self.slot_in_sys_his = slot_in_sys_his
		self.slot_in_usr_his = slot_in_usr_his
		self.slot_request_by_sys = slot_request_by_sys
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
			if example[-5] != 3 and not example[6] and not self.eval:
				p = random.uniform(0,1)
				if p > 0.3:
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
		all_uttr_mask = select_field(data, 'uttr_mask')
		all_val_mask = select_field(data, 'val_mask')
		all_special_tokens_ids = select_field(data, 'special_tokens_ids')

		all_slot_label = [f.slot_label for f in data] # bsz
		all_slot_tag = [f.slot_tag for f in data] # bsz

		all_start = [f.start for f in data] # bsz
		all_end = [f.end for f in data] # bsz
		all_cateVal_idx = [f.cateVal_idx for f in data] # bsz

		all_dialogue_id = [f.dialogue_id for f in data]
		all_api_id = [f.api_id for f in data]
		all_turn_label = [f.turn_label for f in data]
		all_utterance = [f.utterance for f in data]
		all_slot = [f.slot for f in data]
		all_slot_poss_vals = [f.slot_poss_vals for f in data]
		all_slot_in_usr_his = [f.slot_in_usr_his for f in data]
		all_slot_in_sys_his = [f.slot_in_sys_his for f in data]
		all_slot_request_by_sys = [f.slot_request_by_sys for f in data]
		all_uttr_tokens = [f.uttr_tokens for f in data]

		
		input_ids = torch.tensor(all_input_ids, dtype=torch.long)
		input_mask = torch.tensor(all_input_mask, dtype=torch.float)
		segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
		uttr_mask = torch.tensor(all_uttr_mask, dtype=torch.float)
		val_mask = torch.tensor(all_val_mask, dtype=torch.float)
		special_tokens_ids = torch.tensor(all_special_tokens_ids, dtype=torch.long)

		input_ids = input_ids.squeeze(1)
		segment_ids = segment_ids.squeeze(1)
		input_mask = input_mask.squeeze(1)
		uttr_mask = uttr_mask.squeeze(1)
		val_mask = val_mask.squeeze(1)
		special_tokens_ids = special_tokens_ids.squeeze(1)

		slot_label = torch.tensor(all_slot_label, dtype=torch.long)
		start = torch.tensor(all_start, dtype=torch.long)
		end = torch.tensor(all_end, dtype=torch.long)
		cateVal_idx = torch.tensor(all_cateVal_idx, dtype=torch.long)

		dialogue_id = all_dialogue_id
		api_id = all_api_id
		turn_label = all_turn_label

		utterance = all_utterance
		slot = all_slot
		slot_poss_vals = all_slot_poss_vals
		slot_in_usr_his = all_slot_in_usr_his
		slot_in_sys_his = all_slot_in_sys_his
		slot_request_by_sys = all_slot_request_by_sys

		uttr_tokens = all_uttr_tokens
		slot_tag = all_slot_tag


		return [input_ids, input_mask, segment_ids, uttr_mask, val_mask, special_tokens_ids, slot_label, start, end, \
			  cateVal_idx, dialogue_id, api_id, turn_label, utterance, slot, slot_poss_vals, slot_in_usr_his, slot_in_sys_his, \
			  slot_request_by_sys, uttr_tokens, slot_tag]


	def convert_example_to_feature(self, example):
		seq_features = []
		dialogue_id = example[0]
		api_id = example[1]
		turn_label = example[2]
		slot_in_usr_his = example[7]
		slot_in_sys_his = example[8]
		slot_request_by_sys = example[9]
		utterance = example[3]
		slot = example[4]

		uttr_tokens = example[10][-(self.max_uttr_len-1):] + ['null']
		start = example[-3] - max(len(example[10])-self.max_uttr_len+1, 0)
		end = example[-2] - max(len(example[10])-self.max_uttr_len+1, 0)
		slot_tokens = example[11]

		if start < 0:
			start = len(uttr_tokens) - 1
			end = len(uttr_tokens) - 1

		special_tokens = []
		special_tokens_ids = 0

		if slot_in_sys_his:
			special_tokens += ['#', 'in', 'system', 'history']
			special_tokens_ids += 2
		if slot_request_by_sys:
			special_tokens += ['#', 'requested', 'by', 'system']
			special_tokens_ids += 4

		example[12] += [['[PAD]']]*(self.max_numVals_of_slot - len(example[12]))
		example[12] = [['null'], ['do', 'not', 'care']] + example[12]

		assert len(example[12]) == self.max_numVals_of_slot + 2
		assert example[-1] < self.max_numVals_of_slot + 2
		assert example[-1] >= 0

		slot_poss_vals = copy.deepcopy(example[12])

		for i in range(1, len(slot_poss_vals))[::-1]:
			# slot_poss_vals.insert(i, ['#'])
			slot_poss_vals.insert(i, ['[SEP]'])

		for val in slot_poss_vals[1:]:
			slot_poss_vals[0] += val
		slot_poss_vals = slot_poss_vals[0]

		# Modifies `context_tokens_choice` and `ending_tokens` in
		# _truncate_seq_pair(pre_tokens, hyp_tokens, max_seq_length - 3)
		uttr_tokens_padded = uttr_tokens + ['[PAD]']*(self.max_uttr_len-len(uttr_tokens))
		if example[5]:
			tokens = ['[CLS]'] + uttr_tokens_padded + ['[SEP]'] + ['sort', '#'] + slot_tokens \
								+ ['[SEP]'] + slot_poss_vals + ['[SEP]']
		else:
			tokens = ['[CLS]'] + uttr_tokens_padded + ['[SEP]'] + ['tag', '#'] + slot_tokens \
								+ ['[SEP]'] + slot_poss_vals + ['[SEP]']

		segment_ids = [0] * (len(uttr_tokens_padded) + 2) + [1] * (len(slot_tokens) + len(slot_poss_vals) + 4)
		special_tokens_ids = [special_tokens_ids] * self.max_seq_length


		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		# input_mask = [1] * len(input_ids)
		uttr_mask = [1]*len(uttr_tokens)
		uttr_mask += [0]*(self.max_uttr_len - len(uttr_mask))

		val_mask = []
		val_lens = [len(val) for val in example[12]]
		for i in range(len(example[12])):
			# mask_style_1
			tmp_val_mask = [0] * (len(uttr_tokens_padded) + len(slot_tokens)+ 5 + \
				sum(val_lens[:i]) + i) + [1] * val_lens[i]
			tmp_val_mask += [0] * (self.max_seq_length - len(tmp_val_mask))
			val_mask.append(tmp_val_mask)
			
		# manner1
		# uttr_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - len(uttr_tokens)) + \
		# 			[1] * (3 + len(slot_tokens))
		# uttr_att_mask += [0] * (self.max_seq_length - len(uttr_att_mask))

		# manner2
		uttr_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - len(uttr_tokens)) + \
					[1] * (3 + len(slot_tokens) + len(slot_poss_vals))
		uttr_att_mask += [0] * (self.max_seq_length - len(uttr_att_mask))

		slot_att_mask = uttr_att_mask

		cls_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - len(uttr_tokens)) + \
					[1] * (4 + len(slot_tokens) + len(slot_poss_vals))
		cls_att_mask += [0] * (self.max_seq_length - len(cls_att_mask))

		sep1_att_mask = uttr_att_mask
		sep2_att_mask = uttr_att_mask
		sep3_att_mask = cls_att_mask

		input_mask = []
		input_mask.append(cls_att_mask)
		input_mask += [uttr_att_mask] * self.max_uttr_len
		input_mask.append(sep1_att_mask)
		input_mask += [slot_att_mask] * (len(slot_tokens)+2)
		input_mask.append(sep2_att_mask)

		for i in range(len(example[12])):
			# mask_style_1
			tmp_val_att_mask = [1] * (2 + len(uttr_tokens)) + [0] * (self.max_uttr_len - \
									len(uttr_tokens)) + [1] * (3 + len(slot_tokens))
			tmp_val_att_mask += [0] * (sum(val_lens[:i])+i)
			tmp_val_att_mask += [1] * val_lens[i]
			tmp_val_att_mask += [0] * (self.max_seq_length - len(tmp_val_att_mask))

			input_mask += [tmp_val_att_mask] * val_lens[i]
			if i < len(example[12]) -1:
				input_mask.append(cls_att_mask)
		

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
							 segment_ids, uttr_mask, val_mask, special_tokens_ids))

		if self.is_training:
			slot_label = example[-5]
			slot_tag = example[-4][-(self.max_uttr_len-1):] + [0]
			cateVal_idx = example[-1]

		if self.show_example:
			self.show_example = False
			logger.info("*** Example ***")
			logger.info(f"dialog_id: {example[0]}")
			for (tokens, input_ids, input_mask, segment_ids, uttr_mask, val_mask, special_tokens_ids) in seq_features:
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
				slot_tag = slot_tag,
				start = start,
				end = end,
				cateVal_idx = cateVal_idx,
				dialogue_id = dialogue_id,
				api_id = api_id,
				turn_label = turn_label,
				utterance = utterance,
				slot = slot,
				slot_poss_vals = example[12],
				slot_in_usr_his = slot_in_usr_his,
				slot_in_sys_his = slot_in_sys_his,
				slot_request_by_sys = slot_request_by_sys,
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

def calc_loss(logits_cls, logits_start, logits_end, logits_cate, logits_req,\
					 slot_label, start, end, cateVal_idx, uttr_mask):
	loss1 = F.cross_entropy(logits_cls, slot_label)

	shift, _ = logits_start.max(1)
	logits_start = torch.exp(logits_start - shift[:,None]) * uttr_mask
	logits_start = logits_start / logits_start.sum(1)[:,None]
	logits_start = -1 * torch.log(logits_start + 1e-09)

	shift, _ = logits_end.max(1)
	logits_end = torch.exp(logits_end - shift[:,None]) * uttr_mask
	logits_end = logits_end / logits_end.sum(1)[:,None]
	logits_end = -1 * torch.log(logits_end + 1e-09)

	# if (end >= corpus.max_uttr_len).float().sum() > 0 or  (start >= corpus.max_uttr_len).float().sum() > 0:
	# 	print('error1 !')
	# 	exit()

	# if  (cateVal_idx >= corpus.max_numVals_of_slot + 2).float().sum() > 0:
	# 	print('error2 !')
	# 	exit()

	loss2 = torch.gather(logits_start, 1, start[:,None]).squeeze(1).mean() + \
			torch.gather(logits_end, 1, end[:,None]).squeeze(1).mean()
	
	loss3 = F.cross_entropy(logits_cate, cateVal_idx)

	req_label = (slot_label != 2).long()
	loss4 = F.cross_entropy(logits_req, req_label)

	loss = (0.*loss1 + 1.*loss2 + 1.*loss3 + 1.* loss4)/1

	return loss

def accuracy(logits_cls, logits_start, logits_end, logits_cate, logits_req, \
				slot_label, start, end, cateVal_idx, uttr_mask):
	temp_cls_accuracy = np.zeros(4)
	temp_cls_examples = np.zeros(4)
	_, cls_preds = logits_cls.max(1)
	for i in range(4):
		temp_cls_accuracy[i] = ((cls_preds == slot_label) * (slot_label == i)) \
								.float().cpu().numpy().sum()
		temp_cls_examples[i] = (slot_label == i).float().cpu().numpy().sum()

	cls_preds = list(cls_preds.cpu().numpy())
	cls_ground = list(slot_label.cpu().numpy())

	temp_tag_accuracy = np.zeros(2)
	temp_tag_examples = np.zeros(2)

	shift, _ = logits_start.max(1)
	logits_start = torch.exp(logits_start - shift[:,None]) * uttr_mask
	logits_start = logits_start / logits_start.sum(1)[:,None]
	shift, _ = logits_end.max(1)
	logits_end = torch.exp(logits_end - shift[:,None]) * uttr_mask
	logits_end = logits_end / logits_end.sum(1)[:,None]

	
	end_score = logits_end[:,None,:] * triu_mat[None,:,:]
	end_score, end_idx = end_score.max(2)

	over_score = logits_start + end_score
	span_score, start_idx = over_score.max(1)
	end_idx = torch.gather(end_idx, 1, start_idx[:,None]).squeeze(1)

	span_score = list(span_score.detach().cpu().numpy())
	# _, start_idx = logits_start.max(1)
	# _, end_idx = logits_end.max(1)

	uttr_len = uttr_mask.sum(1)

	neg_examples = (start == (uttr_len-1).long()) * (end == (uttr_len-1).long())
	temp_tag_examples[1] = neg_examples.float().sum().cpu().numpy()
	temp_tag_examples[0] = uttr_mask.size(0) - temp_tag_examples[1]


	pred_acc = (start_idx == start) * (end_idx == end)
	temp_tag_accuracy[1] = (neg_examples * pred_acc).float().sum().cpu().numpy()
	temp_tag_accuracy[0] = ((1. - neg_examples) * pred_acc).float().sum().cpu().numpy()
	
	tag_ground = list(torch.stack([start, end], 1).cpu().numpy())
	tag_preds = list(torch.stack([start_idx, end_idx], 1).cpu().numpy())

	
	temp_cate_accuracy = np.zeros(2)
	temp_cate_examples = np.zeros(2)
	_, cate_preds = logits_cate.max(1)
	cate_preds_acc = (cate_preds == cateVal_idx).float()
	temp_cate_accuracy[0] = (cate_preds_acc * (cateVal_idx > 0).float()).cpu().numpy().sum()
	temp_cate_examples[0] = (cateVal_idx > 0).float().cpu().numpy().sum()
	temp_cate_accuracy[1] = (cate_preds_acc * (cateVal_idx == 0).float()).cpu().numpy().sum()
	temp_cate_examples[1] = (cateVal_idx == 0).float().cpu().numpy().sum()

	cate_preds = list(cate_preds.cpu().numpy())
	cate_ground = list(cateVal_idx.cpu().numpy())
	cate_score, _ = F.softmax(logits_cate, 1).max(1)
	cate_score = list(cate_score.detach().cpu().numpy())

	temp_req_accuracy = np.zeros(2)
	temp_req_examples = np.zeros(2)
	_, req_preds = logits_req.max(1)
	req_probs = F.softmax(logits_req, 1)
	req_label = (slot_label != 2).long()
	for i in range(2):
		temp_req_accuracy[i] = ((req_preds == req_label) * (req_label == i)) \
								.float().cpu().numpy().sum()
		temp_req_examples[i] = (req_label == i).float().cpu().numpy().sum()

	req_preds = list(req_preds.cpu().numpy())
	req_ground = list(req_label.cpu().numpy())
	req_score = list(req_probs[:,0].detach().cpu().numpy())

	return temp_cls_accuracy, temp_cls_examples, \
			temp_tag_accuracy, temp_tag_examples, \
			temp_cate_accuracy, temp_cate_examples,\
			temp_req_accuracy, temp_req_examples,\
			cls_preds, tag_preds, cate_preds, req_preds, \
			cls_ground, tag_ground, cate_ground, req_ground, \
			span_score, cate_score, req_score


def train(train_features, epoch, global_step):

	logger.info("\ntraining epoch %d, Num examples = %d, Batch size = %d, Num steps = %d" \
		%(epoch, len(train_features), args.train_batch_size, num_train_steps))

	model.train()
	# for _ in trange(int(args.num_train_epochs), desc="Epoch"):
	tr_loss = 0
	tr_cls_accuracy = np.zeros(4)
	nb_cls_examples = np.zeros(4)

	tr_tag_accuracy = np.zeros(2)
	nb_tag_examples = np.zeros(2)

	tr_cate_accuracy = np.zeros(2)
	nb_cate_examples = np.zeros(2)

	tr_req_accuracy = np.zeros(2)
	nb_req_examples = np.zeros(2)
	nb_tr_steps = 0

	print_steps = 200
	tmp_cls_accuracy = np.zeros(4)
	tmp_cls_examples = np.zeros(4)
	tmp_tag_accuracy = np.zeros(2)
	tmp_tag_examples = np.zeros(2)
	tmp_cate_accuracy = np.zeros(2)
	tmp_cate_examples = np.zeros(2)
	tmp_req_accuracy = np.zeros(2)
	tmp_req_examples = np.zeros(2)

	tmp_loss = 0

	# for step, batch in enumerate(tqdm(batchify(train_features, batch_size=args.train_batch_size),\
	#  		total=num_train_steps, desc="Iteration", miniters=1000, mininterval=100)):
	# for step, batch in enumerate(batchify(train_features, batch_size=args.train_batch_size)):
	for step, batch in enumerate(train_features):
		batch[:10] = tuple(t.to(device) for t in batch[:10])
		input_ids, input_mask, segment_ids, uttr_mask, val_mask, special_tokens_ids, \
				slot_label, start, end, cateVal_idx = batch[:10]

		bsz = input_ids.size(0)

		logits_cls, logits_start, logits_end, logits_cate, logits_req = model(input_ids, segment_ids, input_mask, \
														uttr_mask, val_mask, special_tokens_ids)

		loss = calc_loss(logits_cls, logits_start, logits_end, logits_cate, logits_req, slot_label, \
												start, end, cateVal_idx, uttr_mask)

		temp_cls_accuracy, temp_cls_examples, temp_tag_accuracy, temp_tag_examples, \
		temp_cate_accuracy, temp_cate_examples, temp_req_accuracy, temp_req_examples, \
		 = accuracy(logits_cls, logits_start,\
	 		logits_end, logits_cate, logits_req, slot_label, start, end, cateVal_idx, uttr_mask)[:8]


		if n_gpu > 1:
			loss = loss.mean() # mean() to average on multi-gpu.
		if args.fp16 and args.loss_scale != 1.0:
			# rescale loss for fp16 training
			# see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
			loss = loss * args.loss_scale
		if args.gradient_accumulation_steps > 1:
			loss = loss / args.gradient_accumulation_steps
		tr_loss += loss.item()

		nb_cls_examples += temp_cls_examples
		nb_tag_examples += temp_tag_examples
		nb_cate_examples += temp_cate_examples
		nb_req_examples += temp_req_examples

		nb_tr_steps += 1

		tr_cls_accuracy += temp_cls_accuracy
		tr_tag_accuracy += temp_tag_accuracy
		tr_cate_accuracy += temp_cate_accuracy
		tr_req_accuracy += temp_req_accuracy

		tmp_cls_examples += temp_cls_examples
		tmp_tag_examples += temp_tag_examples
		tmp_cate_examples += temp_cate_examples
		tmp_req_examples += temp_req_examples

		tmp_cls_accuracy += temp_cls_accuracy
		tmp_tag_accuracy += temp_tag_accuracy
		tmp_cate_accuracy += temp_cate_accuracy
		tmp_req_accuracy += temp_req_accuracy

		tmp_loss += loss.item()

		if nb_tr_steps%print_steps == 0:
			tmp_cls_accuracy = tmp_cls_accuracy/tmp_cls_examples
			tmp_tag_accuracy = tmp_tag_accuracy/tmp_tag_examples
			tmp_cate_accuracy = tmp_cate_accuracy/tmp_cate_examples
			tmp_req_accuracy = tmp_req_accuracy/tmp_req_examples

			tmp_loss /= print_steps

			logger.info('steps:%d/total_steps:%d, loss:%.3f, cls_acc:%s, tag_acc:%s, cate_acc:%s, req_acc:%s'% \
								(global_step, num_train_steps, tmp_loss, str(tmp_cls_accuracy), \
											str(tmp_tag_accuracy), str(tmp_cate_accuracy), str(tmp_req_accuracy)))
									
			tmp_cls_accuracy = np.zeros(4)
			tmp_cls_examples = np.zeros(4)
			tmp_tag_accuracy = np.zeros(2)
			tmp_tag_examples = np.zeros(2)
			tmp_cate_accuracy = np.zeros(2)
			tmp_cate_examples = np.zeros(2)
			tmp_req_accuracy = np.zeros(2)
			tmp_req_examples = np.zeros(2)
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
		
	tr_cls_accuracy /= nb_cls_examples
	tr_tag_accuracy /= nb_tag_examples
	tr_cate_accuracy /= nb_cate_examples
	tr_req_accuracy /= nb_req_examples

	tr_loss /= nb_tr_steps
		

	return tr_loss, tr_cls_accuracy, tr_tag_accuracy, tr_cate_accuracy, tr_req_accuracy, global_step, \
			nb_cls_examples, nb_tag_examples, nb_cate_examples, nb_req_examples

def evaluate(eval_features):

	model.eval()
	tr_loss = 0
	tr_cls_accuracy = np.zeros(4)
	nb_cls_examples = np.zeros(4)

	tr_tag_accuracy = np.zeros(2)
	nb_tag_examples = np.zeros(2)

	tr_cate_accuracy = np.zeros(2)
	nb_cate_examples = np.zeros(2)

	tr_req_accuracy = np.zeros(2)
	nb_req_examples = np.zeros(2)

	nb_tr_steps = 0

	total = math.ceil(len(eval_features)/args.train_batch_size)
	n = 0

	total_cls_preds = []
	total_tag_preds = []
	total_cate_preds = []
	total_req_preds = []

	total_dialogue_id = []
	total_api_id = []
	total_turn_label = []
	total_utterance = []
	total_uttr_tag = []
	total_slot = []
	total_slot_poss_vals = []
	total_slot_in_usr_his = []
	total_slot_in_sys_his = []
	total_slot_request_by_sys = []
	total_uttr_tokens = []

	total_cls_ground = []
	total_tag_ground = []
	total_cate_ground = []
	total_req_ground = []

	total_cate_score = []
	total_span_score = []
	total_req_score = []

	eval_loss = 0

	# for sample_idx, batch in enumerate(tqdm(batchify(eval_features, \
	# 					batch_size=args.train_batch_size, shuffle=False), \
	# 					total=total,  desc="Iteration", miniters=1000, mininterval=100)):
	# for sample_idx, batch in enumerate(batchify(eval_features, \
	# 					batch_size=args.train_batch_size, shuffle=False)):
	for sample_idx, batch in enumerate(eval_features):
		batch[:10] = tuple(t.to(device) for t in batch[:10])
		input_ids, input_mask, segment_ids, uttr_mask, val_mask, special_tokens_ids, slot_label, start, end, \
		cateVal_idx, dialogue_id, api_id, turn_label, utterance, slot, slot_poss_vals, slot_in_usr_his, slot_in_sys_his, \
			  slot_request_by_sys, uttr_tokens, slot_tag = batch

		bsz = input_ids.size(0)
		
		with torch.no_grad():
			logits_cls, logits_start, logits_end, logits_cate, logits_req = model(input_ids, segment_ids, input_mask, \
								uttr_mask, val_mask, special_tokens_ids)

			loss = calc_loss(logits_cls, logits_start, logits_end, logits_cate, logits_req, \
								slot_label, start, end, cateVal_idx, uttr_mask)

			temp_cls_accuracy, temp_cls_examples, \
			temp_tag_accuracy, temp_tag_examples, \
			temp_cate_accuracy, temp_cate_examples, \
			temp_req_accuracy, temp_req_examples, \
			cls_preds, tag_preds, cate_preds, req_preds, \
			cls_ground, tag_ground, cate_ground, req_ground, span_score, cate_score, req_score = \
			accuracy(logits_cls, logits_start, logits_end, logits_cate, logits_req, slot_label, \
													start, end, cateVal_idx, uttr_mask)

		eval_loss += loss.mean().item()
		nb_cls_examples += temp_cls_examples
		nb_tag_examples += temp_tag_examples
		nb_cate_examples += temp_cate_examples
		nb_req_examples += temp_req_examples

		nb_tr_steps += 1

		tr_cls_accuracy += temp_cls_accuracy
		tr_tag_accuracy += temp_tag_accuracy
		tr_cate_accuracy += temp_cate_accuracy
		tr_req_accuracy += temp_req_accuracy

		total_dialogue_id += dialogue_id
		total_api_id += api_id
		total_turn_label += turn_label
		total_utterance += utterance
		total_uttr_tag += slot_tag
		total_slot += slot
		total_slot_poss_vals += slot_poss_vals
		total_slot_in_usr_his += slot_in_usr_his
		total_slot_in_sys_his += slot_in_sys_his
		total_slot_request_by_sys += slot_request_by_sys
		total_uttr_tokens += uttr_tokens

		total_cls_ground += cls_ground
		total_tag_ground += tag_ground
		total_cate_ground += cate_ground
		total_req_ground += req_ground

		total_cls_preds += cls_preds
		total_tag_preds += tag_preds
		total_cate_preds += cate_preds
		total_req_preds += req_preds

		total_cate_score += cate_score
		total_span_score += span_score
		total_req_score += req_score

	tr_cls_accuracy /= nb_cls_examples
	tr_tag_accuracy /= nb_tag_examples
	tr_cate_accuracy /= nb_cate_examples
	tr_req_accuracy /= nb_req_examples

	eval_loss /= nb_tr_steps

	total_results = (total_dialogue_id, total_api_id, total_turn_label, \
					total_utterance, total_uttr_tokens, total_uttr_tag, \
					total_slot, total_slot_poss_vals, total_slot_in_usr_his, total_slot_in_sys_his, \
					total_slot_request_by_sys, \
					total_req_score, total_span_score, total_cate_score, \
					total_cls_ground, total_cls_preds,\
					total_tag_ground, total_tag_preds, \
					total_cate_ground, total_cate_preds,\
					total_req_ground, total_req_preds)


	return eval_loss, tr_cls_accuracy, tr_tag_accuracy, tr_cate_accuracy, tr_req_accuracy, \
			nb_cls_examples, nb_tag_examples, nb_cate_examples, nb_req_examples, total_results


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
# corpus = Corpus(args, max_uttr_len=96)
# corpus.mix_train_dev()
# pkl.dump(corpus, open('corpus_withMDdev.pkl', 'wb'))
# exit()

corpus = pkl.load(open('corpus_sample0p0Dev.pkl', 'rb'))
train_data, dev_data, test_data = corpus.get_all_set()
corpus.max_uttr_len=72
corpus.check_length(train_data, maxlen = corpus.max_uttr_len)
print('#train:', len(train_data))
print('#dev:', len(dev_data))
print('#test:', len(test_data))
train_features = DataIterator(train_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=False, batch_size=args.train_batch_size, shuffle=True)
dev_features = DataIterator(dev_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=True, batch_size=args.train_batch_size, shuffle=False)
test_features = DataIterator(test_data, corpus.tokenizer_bert, corpus.max_numVals_of_slot, args.max_seq_length,\
	 					True, corpus.max_uttr_len, eval=True, batch_size=args.train_batch_size, shuffle=False)
print('#filtered train:', len(train_features))
print('#filtered dev:', len(dev_features))
print('#filtered test:', len(test_features))

# train_features = convert_examples_to_features(copy.deepcopy(train_data), corpus.tokenizer_bert, \
# 			corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len)
# dev_features = convert_examples_to_features(copy.deepcopy(dev_data), corpus.tokenizer_bert, \
# 			corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len, eval=True)

# test_features = convert_examples_to_features(copy.deepcopy(test_data), corpus.tokenizer_bert, \
# 			corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len, eval=True)


num_train_steps = int(len(train_features) / args.train_batch_size / \
						args.gradient_accumulation_steps * args.num_train_epochs)


#-------------------------------------------------------------------------------
model = Bert_v1(768, 300, n_layer=1, dropout=0.1, device=device)
print(model)
if args.fp16:
	model.half()


if args.history_model_file is not None:
	model_state_dict = torch.load(args.history_model_file)
	model.load_state_dict(model_state_dict, strict=False)
	logger.info('#Finished loading history model file from %s !'%args.history_model_file)

model.to(device)

triu_mat = torch.triu(torch.ones(corpus.max_uttr_len, corpus.max_uttr_len).to(device))

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
	# if epoch > 0:
		# del train_features
		# train_features = convert_examples_to_features(copy.deepcopy(train_data), corpus.tokenizer_bert, \
		# 		corpus.max_numVals_of_slot, args.max_seq_length, True, corpus.max_uttr_len)
	if args.do_train:
		train_features.filt_data()
		# random.shuffle(train_examples)
		tr_loss, tr_cls_accuracy, tr_tag_accuracy, tr_cate_accuracy, tr_req_accuracy, global_step, \
			tr_cls_examples, tr_tag_examples, tr_cate_examples, tr_req_examples = train(train_features, epoch, global_step)
	else:
		tr_cls_examples, tr_tag_examples, tr_cate_examples, tr_req_examples, tr_loss, \
		tr_cls_accuracy, tr_tag_accuracy, tr_cate_accuracy, tr_req_accuracy = 0, 0, 0, 0, 999., 0., 0., 0., 0.

	eval_loss, eval_cls_accuracy, eval_tag_accuracy, eval_cate_accuracy, eval_req_accuracy, \
	eval_cls_examples, eval_tag_examples, eval_cate_examples, eval_req_examples, total_eval_results = evaluate(dev_features)

	# test_loss, test_cls_accuracy, test_tag_accuracy, test_cate_accuracy, test_req_accuracy, \
	# test_cls_examples, test_tag_examples, test_cate_examples, test_req_examples, total_test_results = evaluate(test_features)


	logging.info('\n'+'#'*40)
	logging.info('epoch%d, train_cls_examples:%s, train_tag_examples:%s, train_cate_examples:%s, train_req_examples:%s' \
			%(epoch, str(tr_cls_examples), str(tr_tag_examples), str(tr_cate_examples), str(tr_req_examples)))
	logging.info('epoch%d, train_cls_acc:%s, train_tag_acc:%s, train_cate_acc:%s, train_req_acc:%s' \
			%(epoch, str(tr_cls_accuracy), str(tr_tag_accuracy), str(tr_cate_accuracy), str(tr_req_accuracy)))

	logging.info('-'*45)
	logging.info('epoch%d, dev_cls_examples:%s, dev_tag_examples:%s, dev_cate_examples:%s, dev_req_examples:%s' \
			%(epoch, str(eval_cls_examples), str(eval_tag_examples), str(eval_cate_examples), str(eval_req_examples)))
	logging.info('epoch%d, dev_cls_acc:%s, dev_tag_acc:%s, dev_cate_acc:%s, dev_reg_acc:%s' \
			%(epoch, str(eval_cls_accuracy), str(eval_tag_accuracy), str(eval_cate_accuracy), str(eval_req_accuracy)))

	# logging.info('-'*45)
	# logging.info('epoch%d, test_cls_examples:%s, test_tag_examples:%s, test_cate_examples:%s, test_req_examples:%s' \
	# 		%(epoch, str(test_cls_examples), str(test_tag_examples), str(test_cate_examples), str(test_req_examples)))
	# logging.info('epoch%d, test_cls_acc:%s, test_tag_acc:%s, test_cate_acc:%s, test_req_acc:%s' \
	# 		%(epoch, str(test_cls_accuracy), str(test_tag_accuracy), str(test_cate_accuracy), str(test_req_accuracy)))

	logging.info('#'*40)


now = datetime.now()
strnow = datetime.strftime(now,'%Y-%m-%d_%H_%M_%S_')
eval_results_path = 'detailed_results/'+strnow+'total_eval_results.pkl'
# test_results_path = 'detailed_results/'+strnow+'total_test_results.pkl'

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


