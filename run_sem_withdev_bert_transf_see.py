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

#transformers的各种模型，bert，roberta
from __future__ import absolute_import, division, print_function
import pandas as pd
import argparse
import logging
import os
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader,TensorDataset)
from tensorboardX import SummaryWriter
import time

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification

from transformers import get_linear_schedule_with_warmup, AdamW

from sklearn import metrics
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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
    epoch_loss = 0

    for batch in dataloader:
        with torch.no_grad():
            out2 = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device))#不需要loss，所以label=None
        label_ids = batch[1].to(device)
        logits = out2[0]
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))
        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = batch[1].to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

    acc, report = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss / len(dataloader),acc, report, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="transformers中的模型都可: bert-base-uncased, roberta-base.")
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

    parser.add_argument("--label_list",
                        default=["0", "1", "2", "3", "4"],
                        type=list,
                        help="我自己加的类别标签")
    parser.add_argument("--predict_test_file",
                        default='ntest_sg_label.tsv',
                        type=str)
    parser.add_argument("--log_dir",
                        default="log_dir",
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")


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

    #读数据，生成dataframe
    df_train = pd.read_csv(args.train_file, sep='\t')
    df_dev = pd.read_csv(args.dev_file, sep='\t')
    df_test = pd.read_csv(args.test_file, sep='\t')

    # Load the pretrained Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=5,
                                                               output_attentions=False, output_hidden_states=False)
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=5,
    #                                                            output_attentions=False, output_hidden_states=False)


    model.to(device)

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
    # hack to remove pooler, which is not used# thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    def encode_fn(text_list):
        all_input_ids = []
        for text in text_list:
            input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=128, return_tensors='pt',pad_to_max_length=True)  # 这个长度得改！！！
            all_input_ids.append(input_ids)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids

    criterion = torch.nn.CrossEntropyLoss()#加了torch
    criterion = criterion.to(device)

    if args.do_train:
        # Create the data loader
        train_text_values = df_train['sentence'].values
        all_input_ids = encode_fn(train_text_values)
        labels = df_train['label'].values
        labels = torch.tensor(labels - 1)  # 减一，让标签从0开始
        train_data = TensorDataset(all_input_ids, labels)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,worker_init_fn=seed_worker)  # _init_fn

        dev_text_values = df_dev['sentence'].values
        dall_input_ids = encode_fn(dev_text_values)
        dlabels = df_dev['label'].values
        dlabels = torch.tensor(dlabels - 1)  # 减一，让标签从0开始
        dev_data = TensorDataset(dall_input_ids, dlabels)
        dev_dataloader = DataLoader(dev_data, batch_size=args.dev_batch_size, worker_init_fn=seed_worker)

        num_train_steps = int(
            len(df_train) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # create optimizer and learning rate schedule
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)  # 要重现BertAdam特定的行为，需设置correct_bias = False
        #total_steps = len(train_dataloader) * args.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*num_train_steps), num_training_steps=num_train_steps)#num_warmup_steps不知道

        logger.info("***** Running training *****transformers")
        logger.info("  Num examples = %d", len(df_train))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("***** Running dev *****")
        logger.info("  Num examples = %d", len(df_dev))
        logger.info("  Batch size = %d", args.dev_batch_size)
        with open(output_eval_file, "a") as writer:###
            writer.write("\t\n***** Running training *****transformers\t\n")
            writer.write("  Num examples = %d\t\n" % len(df_train))
            writer.write("  Batch size = %d\t\n" % args.train_batch_size)
            writer.write("  Num steps = %d\t\n" % num_train_steps)
            writer.write("\t\n***** Running dev *****transformers\t\n")
            writer.write("  Num examples = %d\t\n" % len(df_dev))
            writer.write("  Batch size = %d\t\n" % args.dev_batch_size)

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

                model.train()  # 这个位置正确，保证每一个batch都能进入model.train()的模式

                ##传统的训练函数进来一个batch的数据，计算一次梯度，更新一次网络，而这里用了梯度累加（gradient accumulation）
                ##梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加，累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。
                # 梯度累加步骤：1. input output 获取loss：输入文本和标签，通过infer计算得到预测值，计算损失函数
                out1 = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                             labels=batch[1].to(device))
                loss, logits = out1[:2]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale

                # 2.loss.backward() 反向传播，计算当前梯度 2.1 loss regularization
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_steps += 1

                # 2.2 back propagation
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()## 反向传播求解梯度

                # 用于画图和分析的数据
                epoch_loss += loss.item()
                preds = logits.detach().cpu().numpy()
                outputs = np.argmax(preds, axis=1)
                all_preds = np.append(all_preds, outputs)
                label_ids = label_ids.to('cpu').numpy()
                all_labels = np.append(all_labels, label_ids)

                # 3. 多次循环步骤1-2，不清空梯度，使梯度累加在已有梯度上 update parameters of net
                #梯度累加了一定次数后，先optimizer.step() 根据累计的梯度更新网络参数，然后optimizer.zero_grad() 清空过往梯度，为下一波梯度累加做准备
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)#optimizer_grouped_parameters
                    # 梯度裁剪不再在AdamW中了#大于1的梯度将其设为1.0, 以防梯度爆炸。解决神经网络训练过拟合。只在训练的时候使用，在测试的时候不用
                    optimizer.step()## 更新权重参数 # update parameters of net
                    scheduler.step()
                    optimizer.zero_grad()## 梯度清零 # reset gradient
                    global_step += 1
                    #新增dev数据集调参
                    if global_step % args.print_step == 0 and global_step != 0:
                        num_model += 1
                        train_loss = epoch_loss / train_steps
                        train_acc, train_report = classifiction_metric(all_preds, all_labels, args.label_list)
                        dev_loss, dev_acc, dev_report, _ , _ = evaluate(model, dev_dataloader, criterion, device, args.label_list)

                        c = global_step // args.print_step
                        writer.add_scalar("loss/train", train_loss, c)
                        writer.add_scalar("loss/dev", dev_loss, c)

                        writer.add_scalar("acc/train", train_acc, c)
                        writer.add_scalar("acc/dev", dev_acc, c)

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
                            model_to_save = model.module if hasattr(model,'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            early_stop_times = 0
                        else:
                            early_stop_times += 1

        with open(output_eval_file, "a") as writer:###
            writer.write("\t\n***** Ending dev *****transformers\t\n")
            writer.write("  global_step : %d\t\n" % global_step)
            writer.write("  num_model : %d\t\n" % num_model)
            writer.write("  num_bestacc : %d\t\n" % num_bestacc)

    if args.do_eval:
        # dataframe保存带标签的预测文件ntest_label.tsv,格式：id,text,label,predict_label
        df = pd.DataFrame(columns=['text', 'label', 'predict_label'])
        df['text']=df_test['sentence']

        # Create the test data loader
        test_text_values = df_test['sentence'].values
        tall_input_ids = encode_fn(test_text_values)
        tlabels = df_test['label'].values
        tlabels = torch.tensor(tlabels - 1)  # 减一，让标签从0开始
        pred_data = TensorDataset(tall_input_ids,tlabels)
        pred_dataloader = DataLoader(pred_data, batch_size=args.eval_batch_size, worker_init_fn=seed_worker)

        logger.info("***** Running evaluation *****transformers")
        logger.info("  Num examples = %d", len(df_test))
        logger.info("  Batch size = %d", args.eval_batch_size)

        output_eval_file = os.path.join(args.output_dir, "result.txt")
        output_model_file = os.path.join(args.output_dir, "_pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=5,state_dict=model_state_dict,
                                                                   output_attentions=False, output_hidden_states=False)
        # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=5,state_dict=model_state_dict,
        #                                                            output_attentions=False, output_hidden_states=False)

        model.to(device)
        logger.info("Start evaluating")

        print("=======================")
        print("test_total...")
        _,eval_accuracy, eval_report, all_preds, all_labels = evaluate(model, pred_dataloader,criterion, device, args.label_list)

        df['predict_label'] = all_preds
        df['label'] = all_labels
        ntest_sg_label = os.path.join(args.output_dir, args.predict_test_file)
        df.to_csv(ntest_sg_label, sep='\t')

        eval_macro_f1 = eval_report['macro avg']['f1-score']
        result = {'eval_accuracy': eval_accuracy,'eval_macro_f1':eval_macro_f1}

        with open(output_eval_file, "a") as writer:
            writer.write("***** Running evaluation *****transformers\t\n")
            writer.write("  Num examples = %d\t\n" % df.shape[0])
            writer.write("  Batch size = %d\t\n" % args.eval_batch_size)

            logger.info("***** Eval results *****transformers")
            writer.write("\t\n***** Eval results   %s *****transformers\t\n" % (
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")


if __name__ == "__main__":
    main()