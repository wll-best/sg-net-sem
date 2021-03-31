#将tsv文件中的标签都减一，生成_sub1.tsv文件
def label_sub1(f_old,f_new):
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
            gid.append(i)
            text.append(line[0])
            y.append(int(line[1]))
        print(text, y, gid)
        return  text, y, gid

if __name__ == "__main__":
    label_sub1('train.tsv','train_sub1.tsv')
