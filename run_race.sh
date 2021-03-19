python run_race.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_eval \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --learning_rate 8e-6 \
  --output_dir output_path
