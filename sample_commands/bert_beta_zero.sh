python run_glue.py  --model_name_or_path bert_dir --seed 42\
      --output_dir out_dir --task_name snli  --model_type bert\
      --do_eval --max_seq_length 128 --num_train_epochs 3  \
      --overwrite_output_dir --outputfile results/results.csv \
      --do_lower_case  --ib_dim 144 --deterministic --learning_rate 2e-5\
      --do_train --eval_types dev train test --evaluate_after_each_epoch
