"""
The file is used to provide API for Intent prediction.
Author: Houbowei
Date: Friday, April 1, 2022
Time: 10:46:40
Company: Guangzhou Xiaopeng Motors Technology Co Ltd
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer, BertModel
import pickle as pkl
import datasets
from datasets import load_dataset, load_from_disk

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Bert_v1(nn.Module):

    def __init__(self, ninput, nhidden, n_layer=1, dropout=0, device=None):
        super(Bert_v1, self).__init__()
        self.nhidden = nhidden
        self.ninput = ninput
        self.dropout = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.device = device

        # self.bert = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=None)
        self.bert = BertModel.from_pretrained("bert-base-cased")

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


def main():
~
    dataset = load_from_disk("schema_data")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)
    # model = Bert_v1(768, 300, n_layer=1, dropout=0.1, device=device)
    with open('corpus.pkl', 'rb') as f:
        corpus = pkl.load(f)
    train_data, dev_data, test_data = corpus.get_intent_set()

    random_example = random.choice(dev_data)
    utterance = random_example[3]
    utterance_tokens = random_example[4]
    utterance_tokens.insert(0, "[CLS]")
    utterance_tokens.insert(-1, "[SEP]")

    intents = random_example[5]
    intents_description = random_example[6]
    print(utterance)
    print(intents)
    utt = utterance + "".join(intents)
    inputs = tokenizer(utt,
                       padding="max_length",
                       max_length=196,
                       truncation=True)
    print(tokenizer.decode(inputs['input_ids']))


if __name__ == "__main__":
    main()
