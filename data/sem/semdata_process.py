import csv
import random

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



if __name__ == "__main__":
    # chgt('test.tsv','test_t.tsv')
    # chgt('train.tsv', 'train_t.tsv')
    # chgt('dev.tsv', 'dev_t.tsv')
    semdata_split('sem_t.tsv','ntrain.tsv','ndev.tsv','ntest.tsv')