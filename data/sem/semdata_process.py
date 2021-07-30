import csv
import random
from sklearn import metrics
#将数据集中的$T$改成T，生成_t结尾的文件
def chgt(fin,fout):

    total = 1
    with open(fin,'r', encoding='utf-8') as fi,open(fout,'w',encoding='utf-8',newline="") as fo:
        rowes = csv.reader(fi, delimiter='\t')
        new_rowes=csv.writer(fo,delimiter='\t')
        for row in rowes:
            text = row[0]
            label=row[1]
            if "$T$" in text:
                #先整理一句话中多个特殊字符的情况，变成一句话一个特殊字符
                tnum=text.count("$T$")
                if tnum>1:
                    print(total,text)
                new_text=text.replace("$T$","T")
                new_rowes.writerow([new_text,label])
            else:
                new_rowes.writerow([text,label])
            total += 1
        print(total)


def split(all_list, shuffle=False):
    num = len(all_list)
    if shuffle:
        random.shuffle(all_list)  # 列表随机排序
    train = all_list[:int(num*0.6)]
    dev = all_list[int(num*0.6):int(num*0.8)]
    test = all_list[int(num*0.8):]
    return train, dev,test

#重新划分数据集，打乱后按比例6：2：2分成三个数据集
def semdata_split(fin,ftrain,fdev,ftest):
    with open(fin, 'r', encoding='utf_8') as fi,open(ftrain, 'w', encoding='utf_8',newline="") as ftr,\
            open(fdev, 'w', encoding='utf_8',newline="") as fd,open(ftest, 'w', encoding='utf_8',newline="") as fte:
        reader = csv.reader(fi, delimiter="\t")
        train_rowes=csv.writer(ftr,delimiter='\t')
        dev_rowes=csv.writer(fd,delimiter='\t')
        test_rowes=csv.writer(fte,delimiter='\t')
        lines = []
        for line in reader:
            lines.append(line)
        traindatas, devdatas, testdatas = split(lines, shuffle=True)
        for traindata in traindatas:
            train_rowes.writerow(traindata)
        for devdata in devdatas:
            dev_rowes.writerow(devdata)
        for testdata in testdatas:
            test_rowes.writerow(testdata)


def ntest_label_split(bdf_14,bdf_15,bdf_16,ntest):
    # 找出ntest_label.tsv在14，15，16的bdf.txt中是否存在，如果存在就保存在相应的列表中，最终分别计算三年的准确率
    f14_li=[]
    f15_li=[]
    f16_li=[]
    with open(bdf_14,'r',encoding='utf-8') as f14,open(bdf_15,'r',encoding='utf-8') as f15,\
        open(bdf_16,'r',encoding='utf-8') as f16:
        for line14 in f14.readlines():
            text_all14 = line14.split('####')
            text14 = text_all14[0].strip()
            f14_li.append(text14.replace("$T$","T") if "$T$" in text14 else text14)
        #print('14内容读入:'+str(len(f14_li))+'行')
        #print(f14_li[0:20])
        for line15 in f15.readlines():
            text_all15 = line15.split('####')
            text15 = text_all15[0].strip()
            f15_li.append(text15.replace("$T$","T") if "$T$" in text15 else text15)
        #print('15内容读入:'+str(len(f15_li))+'行')
        for line16 in f16.readlines():
            text_all16 = line16.split('####')
            text16 = text_all16[0].strip()
            f16_li.append(text16.replace("$T$","T") if "$T$" in text16 else text16)
        #print('16内容读入:' + str(len(f16_li)) + '行')
    with open(ntest,'r',encoding='utf-8') as nt:
        next(nt)#跳过第一行
        reader = csv.reader(nt, delimiter="\t")
        lines = []
        row14=[]
        row15=[]
        row16=[]
        rightnum14 = 0
        rightnum15 = 0
        rightnum16 = 0
        target_li_14=[]
        out_li_14=[]
        target_li_15=[]
        out_li_15=[]
        target_li_16=[]
        out_li_16=[]
        for line in reader:
            lines.append(line)
            if line[1] in f14_li:#如果该文本在14中
                row14.append(line)
                target_li_14.append(line[2])
                out_li_14.append(line[3])
                if line[2]==line[3]:
                    rightnum14+=1
            if line[1] in f15_li:#如果该文本在15中
                row15.append(line)
                target_li_15.append(line[2])
                out_li_15.append(line[3])
                if line[2]==line[3]:
                    rightnum15+=1
            if line[1] in f16_li:#如果该文本在16中
                row16.append(line)
                target_li_16.append(line[2])
                out_li_16.append(line[3])
                if line[2]==line[3]:
                    rightnum16+=1

        acc14 = rightnum14 / len(row14)
        acc15 = rightnum15 / len(row15)
        acc16 = rightnum16 / len(row16)
        macro_f1_14 = metrics.f1_score(target_li_14, out_li_14, labels=[0, 1, 2, 3, 4],average='macro')
        macro_f1_15 = metrics.f1_score(target_li_15, out_li_15, labels=[0, 1, 2, 3, 4],average='macro')
        macro_f1_16 = metrics.f1_score(target_li_16, out_li_16, labels=[0, 1, 2, 3, 4],average='macro')

        print('14正确的数目' + str(rightnum14) + ',  14总数目' + str(len(row14)) + ',  14的准确率：' + str(acc14)+',  14的macro_f1：' + str(macro_f1_14))
        print('15正确的数目' + str(rightnum15) + ',  15总数目' + str(len(row15)) + ',  15的准确率：' + str(acc15)+',  15的macro_f1：' + str(macro_f1_15))
        print('16正确的数目' + str(rightnum16) + ',  16总数目' + str(len(row16)) + ',  16的准确率：' + str(acc16)+',  16的macro_f1：' + str(macro_f1_16))


from nltk import word_tokenize
import json
def chg_lal(input_file,out_file):
    #将text_tokens中的元素用word_tokenize重新分割(后缀0删掉)，因为句法树就用了这个----这个函数没用啦
    input_tag_data = []
    with open(input_file,'r') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    guid_to_tag_idx_map = {}
    token_text=[]
    pred_head=[]
    pred_type=[]
    hpsg_list=[]
    for idx, tag_data in enumerate(input_tag_data):
        guid = tag_data["guid"]
        guid_to_tag_idx_map[guid] = idx
        tag_rep = tag_data["tag_rep"]
        token_text.append(word_tokenize(" ".join(tag_rep['text_tokens'])))#不对啊。会把'....'分割成'...'和’.’
        pred_head.append(tag_rep["pred_head_text"])
        pred_type.append(tag_rep["pred_type_text"])
        hpsg_list.append(tag_rep["hpsg_list_text"])

    total = 1
    with open(out_file, 'w') as fout:
        for i in range(len(token_text)):
            data = {}
            data['guid'] = int(total)
            indict = {'text_tokens': token_text[i],
                      'pred_head_text': pred_head[i],
                      'pred_type_text': pred_type[i],
                      'hpsg_list_text': hpsg_list[i]}
            data['tag_rep'] = indict
            fidata = json.dumps(data)
            fout.write(fidata + '\n')
            total += 1

def find_not_eq(input):
    #找长度不等的
    input_tag_data = []
    with open(input,'r') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    guid_to_tag_idx_map = {}
    token_text=[]
    pred_head=[]
    pred_type=[]
    hpsg_list=[]
    for idx, tag_data in enumerate(input_tag_data):
        guid = tag_data["guid"]
        guid_to_tag_idx_map[guid] = idx
        tag_rep = tag_data["tag_rep"]
        #token_text.append(word_tokenize(" ".join(tag_rep['text_tokens'])))
        token_text.append((tag_rep['text_tokens']))
        pred_head.append(tag_rep["pred_head_text"])
        pred_type.append(tag_rep["pred_type_text"])
        hpsg_list.append(tag_rep["hpsg_list_text"])
    wn=0
    for i in range(len(token_text)):
        if len(token_text[i])!=len(hpsg_list[i]):
            wn+=1
            print(token_text[i],len(token_text[i]))
            print(hpsg_list[i],len(hpsg_list[i]))
            print(pred_head[i],len(pred_head[i]))
            print(pred_type[i],len(pred_type[i]))
    print('wrong number'+str(wn))


def roc_sem(labelfile,logitsfile):
    # 引入必要的库
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    #from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    #from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    #from sklearn.multiclass import OneVsRestClassifier
    from numpy import interp

    # 加载数据
    lines = []
    true_label_li = []
    #predict_label_li = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            lines.append(line)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            true_label_li.append(line[2])
            #predict_label_li.append(line[3])
        #print(true_label_li)
    true_label_array = np.array(true_label_li)#List转numpy.array

    # 将标签二值化#!!!cnn:['1','2','3','4','5']-----bert:['0','1','2','3','4']!!!
    y_test = label_binarize(true_label_array, classes=['0','1','2','3','4'])
    #predict_label_array = np.array(predict_label_li)#List转numpy.array

    # 设置种类
    n_classes = y_test.shape[1]

    # 训练模型并预测
    y_score = np.genfromtxt(logitsfile, delimiter=' ', dtype=None)# 将logits.txt文件读入numpy数组

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        #print(i,y_test[:, i], y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve ---  ')# RoBERTa BERT Syntax-BERT BERT TextCNN BiLSTM BiGRU
    plt.legend(loc="lower right")
    plt.show()

def roc_sem_micro(labelfile,logitsfile,classes):
    # 引入必要的库
    import numpy as np

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # 加载数据
    lines = []
    true_label_li = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            lines.append(line)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            true_label_li.append(line[2])
    true_label_array = np.array(true_label_li)#List转numpy.array
    # 将标签classes二值化#!!!cnn:['1','2','3','4','5']-----bert:['0','1','2','3','4']!!!
    y_test = label_binarize(true_label_array, classes)
    # 设置种类
    n_classes = y_test.shape[1]
    # 训练模型并预测
    y_score = np.genfromtxt(logitsfile, delimiter=' ', dtype=None)# 将logits.txt文件读入numpy数组
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr["micro"], tpr["micro"],roc_auc["micro"]

def plt_roc_sem_micro(labelfile1,logitsfile1,labelfile2,logitsfile2,labelfile3,logitsfile3):
    # RoBERTa BERT Syntax-BERT BERT
    import matplotlib.pyplot as plt
    # Plot all ROC curves
    fpr1, tpr1,roc_auc1 = roc_sem_micro(labelfile1,logitsfile1,['0','1','2','3','4'])
    fpr2, tpr2, roc_auc2 = roc_sem_micro(labelfile2, logitsfile2,['0','1','2','3','4'])
    fpr3, tpr3, roc_auc3 = roc_sem_micro(labelfile3, logitsfile3,['0','1','2','3','4'])
    lw = 1.5
    plt.figure()
    plt.plot(fpr1, tpr1,
             label=' BERT (area = {0:0.2f})'
                   ''.format(roc_auc1),
             color='cornflowerblue', linestyle=':', linewidth=3)
    plt.plot(fpr2, tpr2,
             label='RoBERTa (area = {0:0.2f})'
                   ''.format(roc_auc2),
             color='darkorange', linestyle=':', linewidth=3)
    plt.plot(fpr3, tpr3,
             label='Syntax-BERT (area = {0:0.2f})'
                   ''.format(roc_auc3),
             color='red', linestyle=':', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)#反对角线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average ROC curve')# RoBERTa BERT Syntax-BERT BERT
    plt.legend(loc="lower right")
    plt.show()

def plt_roc_sem_micro2(labelfile1,logitsfile1,labelfile2,logitsfile2,labelfile3,logitsfile3,labelfile4,logitsfile4):
    # TextCNN BiLSTM BiGRU
    import matplotlib.pyplot as plt
    # Plot all ROC curves
    fpr1, tpr1,roc_auc1 = roc_sem_micro(labelfile1,logitsfile1,['1','2','3','4','5'])
    fpr2, tpr2, roc_auc2 = roc_sem_micro(labelfile2, logitsfile2,['1','2','3','4','5'])
    fpr3, tpr3, roc_auc3 = roc_sem_micro(labelfile3, logitsfile3,['1','2','3','4','5'])
    fpr4, tpr4, roc_auc4 = roc_sem_micro(labelfile4, logitsfile4,['0','1','2','3','4'])
    lw = 1.5
    plt.figure()
    plt.plot(fpr1, tpr1,
             label='CNN (area = {0:0.2f})'
                   ''.format(roc_auc1),
             color='forestgreen', linestyle=':', linewidth=3)
    plt.plot(fpr2, tpr2,
             label='BiLSTM (area = {0:0.2f})'
                   ''.format(roc_auc2),
             color='chocolate', linestyle=':', linewidth=3)
    plt.plot(fpr3, tpr3,
             label='BiGRU (area = {0:0.2f})'
                   ''.format(roc_auc3),
             color='plum', linestyle=':', linewidth=3)
    plt.plot(fpr4, tpr4,
             label='Syntax-BERT (area = {0:0.2f})'
                   ''.format(roc_auc4),
             color='red', linestyle=':', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)#反对角线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average ROC curve')# TextCNN BiLSTM BiGRU
    plt.legend(loc="lower right")
    plt.show()

def roc_iris():
    # 引入必要的库
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from numpy import interp

    # 加载数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 将标签二值化
    y = label_binarize(y, classes=[0, 1, 2])
    # 设置种类
    n_classes = y.shape[1]

    # 训练模型并预测
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        #print(i, len(y_test[:, i]), len(y_score[:, i]))
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #exit()
    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

'''
数据集说明：
#没打乱顺序
train/test/dev.tsv-->$T$
train_t/test_t/dev_t.tsv-->T
sem_t是所有数据，顺序排列
#打乱顺序按照6：2：2划分
ntrain/ntest/ndev.tsv-->T
ntest_label-->T,模型生成的测试集结果
ntrainss,ntestss-->T,将前面的训练集与验证集整合成现在的训练集，测试集不变
lal_1是用到的句法树
'''

if __name__ == "__main__":
    # chgt('test.tsv','test_t.tsv')
    # chgt('train.tsv', 'train_t.tsv')
    # chgt('dev.tsv', 'dev_t.tsv')
    #semdata_split('sem_t.tsv','ntrain.tsv','ndev.tsv','ntest.tsv')

    # ntest_label_split('Restaurants_All_14_bdf.txt','Restaurants_All_15_bdf.txt','Restaurants_All_16_bdf.txt',
    #                   'F:/01myex/bert_pi8/ntest_sg_label (10).tsv')#bert7e-5
    # ntest_label_split('Restaurants_All_14_bdf.txt','Restaurants_All_15_bdf.txt','Restaurants_All_16_bdf.txt',
    #                   'F:/01myex/roberta_sgbert/ntest_sg_label.tsv')#roberta3e-5
    # ntest_label_split('Restaurants_All_14_bdf.txt','Restaurants_All_15_bdf.txt','Restaurants_All_16_bdf.txt',
    #                   'F:/01myex/33开头/33-1abp/ntest_sg_label55.tsv')#sg5e-5
    #chg_lal('lal_sgnet_ntrain_0.json','lal_sgnet_ntrain.json')
    # find_not_eq('lal_sgnet_ntrain_1.json')
    # print('-------dev--------')
    # find_not_eq('lal_sgnet_ndev_1.json')
    # print('-------test--------')
    # find_not_eq('lal_sgnet_ntest_1.json')
    #roc_iris()
    #改表头-------
    #每种模型的micro-roc和macro-roc
    #roc_sem('F:/01myex/bert_pi8/ntest_sg_label (10).tsv','F:/01myex/bert_pi8/all_logits_bert (10).txt')#bert
    #roc_sem('F:/01myex/roberta_sgbert/ntest_sg_label_rob.tsv','F:/01myex/roberta_sgbert/all_logits_rob.txt')#roberta
    #roc_sem('F:/01myex/33开头/33-1abp/ntest_sg_label55.tsv','F:/01myex/33开头/33-1abp/all_logits_sg55.txt')#sg
    #以上三种模型的micro-roc
    # labelfile1='semresult/ntest_sg_label (10).tsv'
    # logitsfile1='semresult/all_logits_bert (10).txt'
    # labelfile2='semresult/ntest_sg_label_rob.tsv'
    # logitsfile2='semresult/all_logits_rob.txt'
    # labelfile3='semresult/ntest_sg_label55.tsv'
    # logitsfile3='semresult/all_logits_sg55.txt'
    #图片存于semresult/microROC_bert等3个.png
    # plt_roc_sem_micro(labelfile1,logitsfile1,labelfile2,logitsfile2,labelfile3,logitsfile3)


    #roc_sem('F:/a_new_study/lunwen_study/TextClassification/Text-Classification-Models-Pytorch_cnn/Text-Classification-Models-Pytorch/data/sem/ntest_bigru_label.tsv','F:/a_new_study/lunwen_study/TextClassification/Text-Classification-Models-Pytorch_cnn/Text-Classification-Models-Pytorch/data/sem/all_logits_bigru.txt')#cnn
    #roc_sem('F:/01myex/ntest_cnn_label.tsv','F:/01myex/all_logits_cnn.txt')
    #roc_sem('F:/01myex/ntest_bilstm_label.tsv', 'F:/01myex/all_logits_bilstm.txt')
    #roc_sem('F:/01myex/ntest_bigru_label.tsv', 'F:/01myex/all_logits_bigru.txt')
    #以上三种模型+ syntax-bert 的micro-roc
    labelfile1='semresult/ntest_cnn_label.tsv'
    logitsfile1='semresult/all_logits_cnn.txt'
    labelfile2='semresult/ntest_bilstm_label.tsv'
    logitsfile2='semresult/all_logits_bilstm.txt'
    labelfile3='semresult/ntest_bigru_label.tsv'
    logitsfile3='semresult/all_logits_bigru.txt'
    labelfile4='semresult/ntest_sg_label55.tsv'
    logitsfile4='semresult/all_logits_sg55.txt'
    #图片存于semresult/microROC_cnn等4个.png
    plt_roc_sem_micro2(labelfile1,logitsfile1,labelfile2,logitsfile2,labelfile3,logitsfile3,labelfile4,logitsfile4)