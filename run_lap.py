# 读取C:\Users\兰兰\Desktop\Laptop_Implicit_Annotated.txt的隐式方面数据集。
from pandas import DataFrame
from collections import Counter#列表中重复元素统计

def find_category(datafile):
    #df = DataFrame(columns=['text', 'category0', 'category1', 'category2', 'category3', 'category4'])  # ,'lemmatised'
    f = open(datafile, 'r')
    txt = f.readline()
    implicit_set=[]
    #含有[t]的那一行是标题，不要了
    while txt != '':
        #原格式为： 【方面词1】【方面词2】##一句话
        split = txt.split('##')
        #strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        iaspects=split[0]

        if iaspects != '':
            if '][' in iaspects:
                #多个隐式方面
                iaspects_mul=iaspects.strip('[]').split('][')
                #print([iaspects_mul[i] for i in range(0,len(iaspects_mul))])
                for i in range(0, len(iaspects_mul)):
                    implicit_set.append(iaspects_mul[i])
            else:
                implicit_set.append(iaspects.strip('[]锘縖'))#文件开头有‘锘縖’
        txt = f.readline()
    f.close()
    c=Counter(implicit_set)
    print(c)
    print(len(c))

if __name__ == "__main__":
    find_category('C:/Users/兰兰/Desktop/Laptop_Implicit_Annotated_mine.txt')
    #mine版本改正错误的：Perfromance、Performane、Cusotmer Service、Repare（Repair）;大小写统一：Customer service
    '''
    Counter({'Laptop': 273, 'Speed': 26, 'Customer Service': 25, 'Price': 21, 'Heat': 20, 'Size': 19, 'Weight': 13, 'Usage': 13, 'OS': 11,
     'Look': 9, 'Performance': 8, 'Durability': 7, 'Freeze': 5, 'Company': 4, 'Mac': 3, 'Usability': 2, 'Repair': 2, 'Sound': 2, 'Battery': 2,
      'Email': 2, 'Apple': 1, 'Screen': 1, 'keypad': 1, 'Web': 1, 'HP': 1, 'wal mart': 1, 'weight': 1, 'Feel': 1, 'Running': 1, 'Connectivity': 1,
       'View': 1, 'Updates': 1, 'Duarability': 1})
    33
    
    '''