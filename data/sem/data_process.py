'''
do_train
原始文件train.tsv,  tag文件sem_span_train.json
do_eval
原始文件test.tsv,   tag文件sem_span_train.json

原始文件的每行格式sentence'\t'label。注意！！！！！需要加一个gid编号。
标注文件的格式{"guid","tag_rep":{text_tokens,pred_head_text,pred_type_text,hpsg_list_text}}。
'''

from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
def convert_head_to_span(all_heads):
    hpsg_lists = []
    for heads in all_heads:
        n = len(heads)
        childs = [[] for i in range(n + 1)]
        left_p = [i for i in range(n + 1)]
        right_p = [i for i in range(n + 1)]

        def dfs(x):
            for child in childs[x]:
                dfs(child)
                left_p[x] = min(left_p[x], left_p[child])
                right_p[x] = max(right_p[x], right_p[child])

        for i, head in enumerate(heads):
            childs[head].append(i + 1)

        dfs(0)
        hpsg_list = []
        for i in range(1, n + 1):
            hpsg_list.append((left_p[i], right_p[i]))

        hpsg_lists.append(hpsg_list)

    return hpsg_lists

import csv
import json
#注意：先把每个文件里的第一行去掉!!!
def sem_tsv2json(input_file,output_file):

    total = 1
    text_tokens=["Some", "people"]#'句子分隔开'
    pred_head_text=["2", "3"]#hpsg生成的
    pred_type_text=["det", "nsubj"]#hpsg生成的
    hpsg_list_text=["(1, 1)", "(1, 2)"]#例子，convert_head_to_span函数生成的
    with open(input_file, 'r', encoding='utf-8') as fh,open(output_file, 'w', encoding='utf-8') as f:
        rowes = csv.reader(fh, delimiter='\t')
        for row in rowes:
            label = int(row[1])
            text = row[0]
            #生成多层json结构的实现。
            info={}
            data=json.loads(json.dumps(info))
            data['guid']=int(total)
            indict={'text_tokens': text_tokens,
                'pred_head_text': pred_head_text,
                'pred_type_text': pred_type_text,
                'hpsg_list_text': hpsg_list_text}
            data['tag_rep']=indict
            fidata=json.dumps(data)
            f.write(fidata+'\n')
            total += 1

if __name__ == "__main__":
    sem_tsv2json('mytest.tsv','mytest.json')