python run_glue.py  --model_name_or_path bert_path\
       --output_dir out_dir --task_name snli  --model_type bert\
       --do_eval --max_seq_length 128  --num_train_epochs 3   --overwrite_output_dir\
       --outputfile results/results.csv  --do_lower_case  --eval_types train dev test\
       --learning_rate 2e-5 --per_gpu_train_batch_size 8 --do_train --sample_train --num_samples 200\
       --eval_types train test dev --evaluate_after_each_epoch --seed 42
