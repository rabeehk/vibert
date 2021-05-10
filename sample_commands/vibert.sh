python run_glue.py  --model_name_or_path  bert_path  \
    --output_dir out_dir  --task_name snli --model_type bert  --do_eval\
    --max_seq_length 128  --num_train_epochs 25   --overwrite_output_dir \
    --outputfile results/results.csv  --do_lower_case  --ib_dim 384 \
    --beta 1e-05 --ib --learning_rate 2e-5  --do_train --sample_train --num_samples 6000\
    --eval_types dev train test   --kl_annealing linear --evaluate_after_each_epoch --seed 812
