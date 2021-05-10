python run_glue.py  --model_name_or_path bert_path  --output_dir out_dir\
     --task_name snli  --model_type bert  --do_eval --max_seq_length 128\
     --num_train_epochs 3   --overwrite_output_dir --outputfile results/results.csv\
     --do_lower_case  --learning_rate 2e-5 --per_gpu_train_batch_size 32\
     --eval_types train test dev --evaluate_after_each_epoch\
     --seed 42  --do_train  --prior_weight_decay --weight_decay 1e-5
