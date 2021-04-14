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
        rightnum14=0
        rightnum15 = 0
        rightnum16 = 0
        for line in reader:
            lines.append(line)
            if line[1] in f14_li:#如果该文本在14中
                row14.append(line)
                if line[2]==line[3]:
                    rightnum14+=1
            if line[1] in f15_li:#如果该文本在15中
                row15.append(line)
                if line[2]==line[3]:
                    rightnum15+=1
            if line[1] in f16_li:#如果该文本在16中
                row16.append(line)
                if line[2]==line[3]:
                    rightnum16+=1

        acc14 = rightnum14 / len(row14)
        acc15 = rightnum15 / len(row15)
        acc16 = rightnum16 / len(row16)

        print('14正确的数目' + str(rightnum14) + ',  14总数目' + str(len(row14)) + ',  14的准确率：' + str(acc14))
        print('15正确的数目' + str(rightnum15) + ',  15总数目' + str(len(row15)) + ',  15的准确率：' + str(acc15))
        print('16正确的数目' + str(rightnum16) + ',  16总数目' + str(len(row16)) + ',  16的准确率：' + str(acc16))



if __name__ == "__main__":
    # chgt('test.tsv','test_t.tsv')
    # chgt('train.tsv', 'train_t.tsv')
    # chgt('dev.tsv', 'dev_t.tsv')
    #semdata_split('sem_t.tsv','ntrain.tsv','ndev.tsv','ntest.tsv')

    ntest_label_split('Restaurants_All_14_bdf.txt','Restaurants_All_15_bdf.txt','Restaurants_All_16_bdf.txt',
                      'ntest_label.tsv')