# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from __future__ import absolute_import, division, print_function

#我的
import pandas as pd
import json
import time

import argparse
import csv
import logging
import os
import random
import pickle
from tqdm import tqdm, trange
import spacy
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler


from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
from sklearn import metrics

class SemExample(object):
    """A single training/test example for simple sequence classification."""
    #由race里的que对应sem里的text，增加text_heads,text_types,text_span,
    def __init__(self,
                 seid,
                 text_a,  #这个删吗？
                 text_heads,#新增3个
                 text_types,
                 text_span,
                 token_text,
                 text_b=None,  # 这个删吗？
                 label=None):
        self.seid = seid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.text_heads = text_heads
        self.text_types = text_types
        self.text_span = text_span
        self.token_text = token_text

    ###新增2个函数
    def __str__(self):
        """__str__使用：被打印的时候需要以字符串的形式输出的时候，
        就会找到这个方法，并将返回值打印出来"""
        return self.__repr__()
    def __repr__(self):
        """返回一个可以用来表示对象的可打印字符串,
         同时定义 __repr__ 方法和 __str__ 方法时，print() 方法会调用 __str__ 方法."""
        l = [
            f"seid: {self.seid}",
            f"text_a: {self.text_a}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    """A single set of features of data.##新增example_id,choices_features。改函数结构。"""
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'text_len': text_len,###将ques改成text
                'input_span_mask': input_span_mask###
            }
            for _, input_ids, input_mask, segment_ids, text_len, input_span_mask in
            choices_features
        ]
        self.label = label-1

        assert self.label>=0
        assert self.label<=4


def rea_sem(path):
    with open(path, 'r', encoding='utf_8') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        text = []
        y = []
        gid = []
        for line in reader:
            lines.append(line)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            gid.append(i)#这里我把编号从1开始排序
            text.append(line[0])
            y.append(int(line[1]))####改成int型
        return  text, y, gid#返回三个数组：句子数组，标签数组，编号数组

###新增
class SimpleNlp(object):
    def __init__(self):#这得改，要用spacy打标签----------
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        # self.nlp = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

    def nlp(self, text):
        return self.nlp(text)

###新增
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_sem_examples(input_file, input_tag_file, is_training):
    text, y, gid = rea_sem(input_file)

    input_tag_data = []

    simple_nlp = SimpleNlp()
    with open(input_tag_file, "r", encoding='utf-8') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    guid_to_tag_idx_map = {}
    all_dqtag_data = []
    for idx, tag_data in enumerate(tqdm(input_tag_data, ncols=50, desc="tagging...")):
        guid = tag_data["guid"]
        guid_to_tag_idx_map[guid] = idx
        tag_rep = tag_data["tag_rep"]
        dqtag_data = {
            "guid": guid,
            "head_text": [int(i) for i in tag_rep["pred_head_text"]],
            "span_text": [eval(i) for i in tag_rep["hpsg_list_text"]],
            "type_text": tag_rep["pred_type_text"],
            "token_text": tag_rep['text_tokens'],###改成双引号
        }
        all_dqtag_data.append(dqtag_data)

    examples = []
    for i, (s1, s2, s3), in enumerate(
            tqdm(zip(text, y, gid), total=len(gid), ncols=50, desc="reading...")):
        dqtag = all_dqtag_data[guid_to_tag_idx_map[s3]]

        assert dqtag["guid"] == s3

        examples.append(
            SemExample(
                seid=s3,
                text_a=s1,
                label=s2 if is_training else None,
                text_heads=dqtag["head_text"],
                text_types=dqtag["type_text"],
                text_span=dqtag["span_text"],
                token_text=dqtag["token_text"]
            )
        )

    return examples

###
def get_sub_spans(que_tokens, que_types, tokenizer, que_span):
    que_org_to_split_map = {}
    pre_tok_len = 0
    sub_que_types = []
    sub_que_span = []
    query_tokens = []

    assert len(que_tokens) == len(que_span)
    for idx, que_token in enumerate(que_tokens):
        sub_que_type_list = [que_types[idx]]
        sub_que_tok = tokenizer.tokenize(que_token)
        query_tokens.extend(sub_que_tok)
        while len(sub_que_type_list) != len(sub_que_tok):
            sub_que_type_list.append("subword")
        sub_que_types.extend(sub_que_type_list)
        que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_que_tok)

    for idx, (start_ix, end_ix) in enumerate(que_span):
        head_start, head_end = que_org_to_split_map[idx]##不对劲
        # sub_start_idx and sub_end_idx of children of head node
        head_spans = [(que_org_to_split_map[start_ix - 1][0], que_org_to_split_map[end_ix - 1][1])]
        # all other head sub_tok point to first head sub_tok
        if head_start != head_end:
            head_spans.append((head_start + 1, head_end))
            sub_que_span.append(head_spans)

            for i in range(head_start + 1, head_end + 1):
                sub_que_span.append([(i, i)])
        else:
            sub_que_span.append(head_spans)

    assert len(sub_que_span) == len(query_tokens)

    return sub_que_span,query_tokens


def convert_examples_to_features(examples, tokenizer, max_seq_length):#, label_list
    """Loads a data file into a list of `InputBatch`s."""
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    #label_map = {label : i for i, label in enumerate(label_list)}#要不要？

    features = []
    for example_index, example in enumerate(tqdm(examples, ncols=50, desc="converting...")):
        text_types=example.text_types
        text_span=example.text_span
        org_text_token = example.token_text
        sub_text_spans,tokens_a = get_sub_spans(org_text_token, text_types, tokenizer, text_span)
        #这里tokens_a的切分做更改了

        # making masks
        text_span_mask = np.zeros((len(sub_text_spans), len(sub_text_spans)))
        for idx, span_list in enumerate(sub_text_spans):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    text_span_mask[start_ix:end_ix + 1, idx] = 1

        #tokens_a = tokenizer.tokenize(example.text_a)
        #assert len(sub_text_spans) == len(tokens_a)

        choices_features = []
        context_text_span_mask = np.zeros((len(tokens_a), len(tokens_a)))
        context_text_span_mask[0:len(tokens_a), 0:len(tokens_a)] = text_span_mask

        # tokens_b = None
        #         # if example.text_b:
        #         #     tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        idxa = list(range(len(tokens_a)))
        idxa2=idxa[0]#0

        input_span_mask = np.zeros((max_seq_length, max_seq_length))
        # 0 count for [CLS] and select_doc_len+1 count for [SEP]
        input_span_mask[1:len(tokens_a) + 1, 1:len(tokens_a) + 1] = context_text_span_mask[idxa2:, idxa2:]###

        record_mask = []
        for i in range(max_seq_length):
            i_mask = []
            for j in range(max_seq_length):
                if input_span_mask[i, j] == 1:
                    i_mask.append(j)
            record_mask.append(i_mask)

        text_len=len(tokens_a)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * (len(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert (text_len) <= max_seq_length

        choices_features.append(
            (tokens, input_ids, input_mask, segment_ids, text_len, record_mask))
        # if example.label == "0":
        #     label_id = 100
        # else:
        #     label_id = label_map[example.label]
        # if example < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        label = example.label
        features.append(
            InputFeatures(
                example_id=example.seid,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    idx_a = list(range(len(tokens_a)))#新增
    idx_b = list(range(len(tokens_b)))#新增
    while True:

        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
            idx_a.pop(0)
        else:
            tokens_b.pop(0)
            idx_b.pop(0)

    return idx_a[0], idx_b[0]

#一样
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)#当axis=1，是在行中比较，选出最大的 列 索引
    return np.sum(outputs == labels)

#新增
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

#一样
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

####新增
def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

####新增
def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def evaluate(model, dataloader,dev_features,device):

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids, example_index in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        # 新增开始,和do_train部分一样
        input_span_mask = np.zeros((input_ids.size(0), input_ids.size(1), input_ids.size(2), input_ids.size(2)))
        for batch_idx, ex_idx in enumerate(example_index):
            dev_feature = dev_features[ex_idx.item()]
            choice_features = dev_feature.choices_features
            for idx, choice_fea in enumerate(choice_features):
                span_mask = choice_fea["input_span_mask"]
                for i, i_mask in enumerate(span_mask):
                    for j in i_mask:
                        input_span_mask[batch_idx, idx, i, j] = 1
        input_span_mask = torch.tensor(input_span_mask, dtype=torch.long)
        # 新增结束
        with torch.no_grad():

            logits = model(input_ids, segment_ids, input_mask, input_span_mask=input_span_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples
    return eval_accuracy

def main():
    parser = argparse.ArgumentParser()

    ## 有用参数
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--output_file",
                        # default='output_batch4_gpu4_large_qo_lamda10_fp16.txt',
                        default='output_file.txt',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_file",
                        default='data/sem/ntrain.tsv',
                        type=str)
    parser.add_argument("--test_file",
                        default='data/sem/ntest.tsv',
                        type=str)
    parser.add_argument("--train_tag_file",
                        default='data/sem/output_sgnet_ntrain.json',
                        type=str)
    parser.add_argument("--test_tag_file",
                        default='data/sem/output_sgnet_ntest.json',
                        type=str)
    parser.add_argument("--dev_file",
                        default='data/sem/ndev.tsv',
                        type=str)
    parser.add_argument("--dev_tag_file",
                        default='data/sem/output_sgnet_ndev.json',
                        type=str)
    parser.add_argument('--n_gpu',
                        type=int, default=2,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",#用uncased无大小写模型时要这个
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
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
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=5,#原来是4
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    #增加dev集

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--print_step",
                        default=100,
                        type=int,
                        help="多少步进行模型保存以及日志信息写入")
    parser.add_argument("--early_stop", type=int, default=50, help="提前终止，多少次dev loss 连续增大，就不再训练")

    # parser.add_argument("--label_list",
    #                     default=["1", "2", "3", "4", "5"],
    #                     type=list,
    #                     help="我自己加的类别标签")#不知道有没有用

    ## 没用参数
    '''
    parser.add_argument("--data_dir",
                        type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")    
    parser.add_argument("--model_name",
                        default='output_batch4_fp16.bin',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dev_file",
                        default='dev.json',
                        type=str)
    parser.add_argument("--dev_tag_file",
                        default='output_sem_dev.json',
                        type=str)
    parser.add_argument("--sa_file",#这是什么
                        default='sample_nsp_train_dev.json',
                        type=str)
    parser.add_argument("--test_middle",
                        default='testmiddle.json',
                        type=str)
    parser.add_argument("--test_high",
                        default='testhigh.json',
                        type=str)
    parser.add_argument("--mid_high",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_sa",#这是什么
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--load_ft",#这是什么
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    '''

    args = parser.parse_args()
    logger.info(args)
    output_eval_file = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_eval_file, "w") as writer:
        writer.write("%s\t\n" % args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = args.n_gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = args.n_gpu
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

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # model = BertForSequenceClassificationSpanMask.from_pretrained(args.bert_model,
    #                                                       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
    #                                                           args.local_rank),
    #                                                       num_labels=5)###要改这个
    model = BertForMultipleChoiceSpanMask2.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_choices=5)###要改这个
    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = read_sem_examples(args.train_file, args.train_tag_file,
                                            is_training=True)###要改这个
        dev_examples = read_sem_examples(args.dev_file, args.dev_tag_file,
                                            is_training=True)###要改这个
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    if args.fp16:
        model.half()

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

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


    if args.do_train:

        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )
        #增加dev_dataloader
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )

        logger.info("***** Running training *****sg-net-sem")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("***** Running dev *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)


        with open(output_eval_file, "a") as writer:###
            writer.write("***** Running training *****\t\n")
            writer.write("  Num examples = %d\t\n" % len(train_examples))
            writer.write("  Batch size = %d\t\n" % args.train_batch_size)
            writer.write("  Num steps = %d\t\n" % num_train_steps)
            writer.write("***** Running dev *****\t\n")
            writer.write("  Num examples = %d\t\n" % len(dev_examples))
            writer.write("  Batch size = %d\t\n" % args.dev_batch_size)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)##新增

        dall_input_ids = torch.tensor(select_field(dev_features, 'input_ids'), dtype=torch.long)
        dall_input_mask = torch.tensor(select_field(dev_features, 'input_mask'), dtype=torch.long)
        dall_segment_ids = torch.tensor(select_field(dev_features, 'segment_ids'), dtype=torch.long)
        dall_label = torch.tensor([f.label for f in dev_features], dtype=torch.long)
        dall_example_index = torch.arange(dall_input_ids.size(0), dtype=torch.long)##新增

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_example_index)
        dev_data = TensorDataset(dall_input_ids, dall_input_mask, dall_segment_ids, dall_label, dall_example_index)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            dev_sampler = RandomSampler(dev_data)
        else:
            train_sampler = DistributedSampler(train_data)
            dev_sampler = DistributedSampler(dev_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

        TrainLoss = []#新增
        global_step = 0
        best_acc = 0
        early_stop_times = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            if early_stop_times >= args.early_stop:
                break

            #tr_loss = 0
            epoch_loss = 0
            nb_tr_examples, train_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, ncols=50, desc="Iteration")):#新增ncols，进度条长度。默认是10

                model.train()  # 这个位置到底在哪

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, example_index = batch#新增example_index
                #新增开始，input_span_mask----race是0，1，2，2四维.squad是0，1，1三维
                input_span_mask = np.zeros((input_ids.size(0), input_ids.size(1), input_ids.size(2), input_ids.size(2)))
                for batch_idx, ex_idx in enumerate(example_index):
                    train_feature = train_features[ex_idx.item()]
                    choice_features = train_feature.choices_features
                    for idx, choice_fea in enumerate(choice_features):
                        train_span_mask = choice_fea["input_span_mask"]
                        for i, i_mask in enumerate(train_span_mask):###race比squad多这个i
                            for j in i_mask:
                                input_span_mask[batch_idx, idx, i, j] = 1
                input_span_mask = torch.tensor(input_span_mask, dtype=torch.long)
                #新增结束---
                loss = model(input_ids, segment_ids, input_mask, label_ids,
                             input_span_mask=input_span_mask)#新增input_span_mask
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # tr_loss += loss.item()#epoch_loss
                # nb_tr_examples += input_ids.size(0)#没用

                train_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    #新增dev数据集调参
                    if global_step % args.print_step == 0 and global_step != 0:
                        train_loss = epoch_loss / train_steps
                        dev_acc = evaluate(model, dev_dataloader, dev_features,device)
                        # 以 acc 取优
                        if dev_acc > best_acc:
                            best_acc = dev_acc
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                            early_stop_times = 0
                        else:
                            early_stop_times += 1

            #---------------------------------------------
        #     TrainLoss.append(epoch_loss/train_steps)#这个咋整
        #
        # with open(os.path.join(args.output_dir, "train_loss.pkl"), 'wb') as f:
        #     pickle.dump(TrainLoss, f)

    if args.do_eval:

        # with open(os.path.join(args.output_dir, "train_loss.pkl"), 'rb') as f:
        #     TrainLoss = pickle.load(f)
        text_li, _, _ = rea_sem(args.test_file)#为了读文本新增这一行
        #dataframe保存带标签的预测文件ntest_label.tsv,格式：id,text,label,predict_label
        df=pd.DataFrame(columns=['text', 'label', 'predict_label'])
        df['text']=text_li
        eval_examples = read_sem_examples(args.test_file, args.test_tag_file,
                                           is_training=True)###要改！！
        total_eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )
            ##label_list=args.label_list)  # label_list要不要呢？先加上吧

        eval_features = total_eval_features
        logger.info("***** Running evaluation *****sg-net-sem")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                  all_example_index)
        eval_sampler = SequentialSampler(eval_data)#和traindata不同的sampler

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        output_eval_file = os.path.join(args.output_dir, "result.txt")

        #for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        #train_loss = TrainLoss[epoch]
        output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        # model = BertForSequenceClassificationSpanMask.from_pretrained(args.bert_model, state_dict=model_state_dict,
        #                                                       num_labels=5)###改
        model = BertForMultipleChoiceSpanMask2.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                              num_choices=5)
        model.to(device)
        logger.info("Start evaluating")

        print("=======================")
        print("test_total...")
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_li=[]
        predict_label_li=[]
        for input_ids, input_mask, segment_ids, label_ids, example_index in eval_dataloader:

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            #新增开始,和do_train部分一样
            input_span_mask = np.zeros((input_ids.size(0), input_ids.size(1), input_ids.size(2), input_ids.size(2)))
            for batch_idx, ex_idx in enumerate(example_index):
                total_eval_feature = total_eval_features[ex_idx.item()]
                choice_features = total_eval_feature.choices_features
                for idx, choice_fea in enumerate(choice_features):
                    span_mask = choice_fea["input_span_mask"]
                    for i, i_mask in enumerate(span_mask):
                        for j in i_mask:
                            input_span_mask[batch_idx, idx, i, j] = 1
            input_span_mask = torch.tensor(input_span_mask, dtype=torch.long)
            #新增结束
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids,
                                      input_span_mask=input_span_mask)
                logits = model(input_ids, segment_ids, input_mask, input_span_mask=input_span_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            tmp_eval_accuracy = accuracy(logits, label_ids)#一个eval_batch中预测对的个数,均为0-4
            
            label_li.append(label_ids.tolist())
            predict_label_li.append(np.argmax(logits,axis=1).tolist())

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1


        df['predict_label']=sum(predict_label_li,[])   
        df['label']=sum(label_li,[])#这个label_li是标签减去1，即索引的列表。sum这个函数是将二维列表变一维列表      
        df.to_csv("ntest_sg_label.tsv",sep='\t')
        macro_f1 = metrics.f1_score(sum(predict_label_li,[]),sum(label_li,[]),labels=[0,1,2,3,4], average='macro')
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'macro_f1':macro_f1}

        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("\t\n***** Eval results   %s *****\t\n" % (
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")


if __name__ == "__main__":
    main()