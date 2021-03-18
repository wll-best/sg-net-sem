'''
do_train
原始文件train.tsv,  tag文件sem_span_train.json
do_eval
原始文件test.tsv,   tag文件sem_span_train.json

原始文件的每行格式sentence'\t'label。注意！！！！！需要加一个gid编号。
标注文件的格式{"guid","tag_rep":{text_tokens,pred_head_text,pred_type_text,hpsg_list_text}}。
'''
