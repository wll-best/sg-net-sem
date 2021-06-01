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
#测试baseline-----原始的bert-sem
from __future__ import absolute_import, division, print_function

#我的
import pandas as pd
import argparse
import csv
import random
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn import metrics
from tensorboardX import SummaryWriter
import time

import torch.nn.functional as F

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id-1


def read_sem_examples(input_file, is_training):
    with open(input_file, 'r', encoding='utf_8') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line1 in reader:
            lines.append(line1)

    if is_training and lines[0][-1] != 'label':
        raise ValueError(
            "For training, the input file must contain a label column."
        )

    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        text_a = line[0]
        label = int(line[1])
        examples.append(
            InputExample(guid=i, text_a=text_a, text_b=None, label=label))

    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):#不增label_list
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


    #label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, ncols=50, desc="converting...")):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #label_id=label_map[example.label]
        label_id = example.label
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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


#一样
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]
    #多分类：micro - F1 = micro - precision = micro - recall = accuracy
    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)
    #digits：int，输出浮点值的位数．

    return acc, report

def evaluate(model, dataloader,criterion, device, label_list):
    model.eval()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_logits = np.array([], dtype=int)#每种分类的可能性数组
    epoch_loss = 0

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))
        preds = logits.detach().cpu().numpy()

        all_logits=np.append(all_logits,F.softmax(logits.detach().cpu(),dim=1))#每种分类的可能性数组

        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss / len(dataloader), acc, report, all_logits, all_preds, all_labels

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
    parser.add_argument("--dev_file",
                        default='data/sem/ndev.tsv',
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
                        type=float, default=4,#原来是4
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    #增加dev集

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--print_step",
                        default=50,
                        type=int,
                        help="多少步进行模型保存以及日志信息写入")
    parser.add_argument("--early_stop", type=int, default=50, help="提前终止，多少次dev acc 不再连续增大，就不再训练")
    parser.add_argument("--log_dir",
                        default="log_dir",
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")
    parser.add_argument("--label_list",
                        default=["0", "1", "2", "3", "4"],
                        type=list,
                        help="我自己加的类别标签")
    parser.add_argument("--predict_test_file",
                        default='ntest_sg_label.tsv',
                        type=str)

    args = parser.parse_args()
    logger.info(args)
    output_eval_file = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)#如果已经存在，不抛出异常

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

    #为了复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现。
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_labels=5)###要改这个
    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = read_sem_examples(args.train_file,is_training=True)
        dev_examples = read_sem_examples(args.dev_file,is_training=True)
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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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

        logger.info("***** Running training *****bert-sem")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("***** Running dev *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)


        with open(output_eval_file, "a") as writer:###
            writer.write("\t\n***** Running training *****bert-sem\t\n")
            writer.write("  Num examples = %d\t\n" % len(train_examples))
            writer.write("  Batch size = %d\t\n" % args.train_batch_size)
            writer.write("  Num steps = %d\t\n" % num_train_steps)
            writer.write("\t\n***** Running dev *****bert-sem\t\n")
            writer.write("  Num examples = %d\t\n" % len(dev_examples))
            writer.write("  Batch size = %d\t\n" % args.dev_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        dall_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        dall_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        dall_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        dall_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
        dev_data = TensorDataset(dall_input_ids, dall_input_mask, dall_segment_ids, dall_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            dev_sampler = SequentialSampler(dev_data)
            #在训练的时候，我们使用的是RandomSampler采样器，在验证或者测试的时候，我们使用的是SequentialSampler采样器。
            # 训练的时候是打乱数据再进行读取，验证的时候顺序读取数据
        else:
            train_sampler = DistributedSampler(train_data)
            dev_sampler = DistributedSampler(dev_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,worker_init_fn=seed_worker)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size,worker_init_fn=seed_worker)

        global_step = 0
        best_acc = 0
        early_stop_times = 0

        writer = SummaryWriter(
            log_dir=args.log_dir + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))
        num_model = 0
        num_bestacc=0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            if early_stop_times >= args.early_stop:
                print('early_stop......')
                break

            print(f'---------------- Epoch: {epoch + 1:02} ----------')

            epoch_loss = 0
            all_preds = np.array([], dtype=int)
            all_labels = np.array([], dtype=int)
            train_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, ncols=50, desc="Iteration")):#新增ncols，进度条长度。默认是10

                model.train()  # 这个位置到底在哪

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #新增结束---
                #loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss=criterion(logits, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # 用于画图和分析的数据
                epoch_loss += loss.item()
                preds = logits.detach().cpu().numpy()
                outputs = np.argmax(preds, axis=1)
                all_preds = np.append(all_preds, outputs)
                label_ids = label_ids.to('cpu').numpy()
                all_labels = np.append(all_labels, label_ids)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    #新增dev数据集调参--global_step是print_step的倍数才执行下面的
                    if global_step % args.print_step == 0 and global_step != 0:
                        num_model += 1
                        train_loss = epoch_loss / train_steps
                        train_acc, train_report = classifiction_metric(all_preds, all_labels, args.label_list)
                        dev_loss, dev_acc, dev_report, _ , _ , _ = evaluate(model, dev_dataloader, criterion, device, args.label_list)

                        c = global_step // args.print_step
                        writer.add_scalar("loss/train", train_loss, c)
                        writer.add_scalar("loss/dev", dev_loss, c)

                        writer.add_scalar("micro_f1/train", train_acc, c)##acc/train
                        writer.add_scalar("micro_f1/dev", dev_acc, c)##acc/dev

                        for label in args.label_list:
                            writer.add_scalar(label + "_" + "f1/train", train_report[label]['f1-score'], c)
                            writer.add_scalar(label + "_" + "f1/dev",
                                              dev_report[label]['f1-score'], c)

                        print_list = ['macro', 'weighted']
                        for label in print_list:
                            writer.add_scalar(label + "_avg_" +"f1/train",
                                              train_report[label+' avg']['f1-score'], c)
                            writer.add_scalar(label + "_avg_" + "f1/dev",
                                              dev_report[label+' avg']['f1-score'], c)

                        # 以 acc 取优
                        if dev_acc > best_acc:
                            num_bestacc += 1
                            best_acc = dev_acc
                            # Save a trained model
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                            early_stop_times = 0
                        else:
                            early_stop_times += 1

        with open(output_eval_file, "a") as writer:###
            writer.write("\t\n***** Ending dev *****bert-sem\t\n")
            writer.write("  global_step : %d\t\n" % global_step)
            writer.write("  num_model : %d\t\n" % num_model)
            writer.write("  num_bestacc : %d\t\n" % num_bestacc)

    writer.close()

    if args.do_eval:
        text_li=[]

        # dataframe保存带标签的预测文件ntest_label.tsv,格式：id,text,label,predict_label
        df = pd.DataFrame(columns=['text', 'label', 'predict_label'])
        eval_examples = read_sem_examples(args.test_file,is_training=True)###要改！！
        for exa in eval_examples:
            text_li.append(exa.text_a)
        df['text']=text_li

        total_eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )

        eval_features = total_eval_features
        logger.info("***** Running evaluation *****bert-sem")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)#和traindata不同的sampler

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,worker_init_fn=seed_worker)

        output_eval_file = os.path.join(args.output_dir, "result.txt")

        output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)

        model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                              num_labels=5)
        model.to(device)
        logger.info("Start evaluating")

        print("=======================")
        print("test_total...")

        _, eval_accuracy, eval_report, all_logits, all_preds, all_labels = evaluate(model, eval_dataloader, criterion, device, args.label_list)

        df['predict_label'] = all_preds
        df['label'] = all_labels
        ntest_sg_label = os.path.join(args.output_dir, args.predict_test_file)
        df.to_csv(ntest_sg_label, sep='\t')

        eval_macro_f1 = eval_report['macro avg']['f1-score']
        result = {'eval_accuracy': eval_accuracy,'eval_macro_f1':eval_macro_f1}

        with open(output_eval_file, "a") as writer:
            writer.write("***** Running evaluation *****bert-sem\t\n")
            writer.write("  Num examples = %d\t\n" % len(eval_examples))
            writer.write("  Batch size = %d\t\n" % args.eval_batch_size)

            logger.info("***** Eval results *****bert-sem")
            writer.write("\t\n***** Eval results   %s *****bert-sem\t\n" % (
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")

        np.savetxt(args.output_dir+'/all_logits_bert.txt', all_logits.reshape(-1,5))

if __name__ == "__main__":
    main()