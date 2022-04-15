# training and validation
python train.py \
--save_total_limit 5 \
--overwrite_output_dir \
--output_dir ./exp \
--logging_dir ./logs \
--PLM gogamza/kobart-base-v2 \
--max_input_length 1024 \
--max_target_length 128 \
--do_train \
--do_eval \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--preprocessing_num_workers 4 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--weight_decay 1e-3 \
--warmup_ratio 0.05 \
--logging_steps 20 \
--save_steps 100 \
--eval_steps 100 \
--evaluation_strategy steps \
--save_strategy steps \
--generation_max_length 128 \
--generation_num_beams 3 

# full training
python train.py \
--save_total_limit 5 \
--overwrite_output_dir \
--output_dir ./exp \
--logging_dir ./logs \
--PLM gogamza/kobart-base-v2 \
--max_input_length 1024 \
--max_target_length 128 \
--do_train \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--preprocessing_num_workers 4 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--weight_decay 1e-3 \
--warmup_ratio 0.05 \
--logging_steps 20 \
--save_steps 100 \
--evaluation_strategy no \
--save_strategy steps

# inference
python inference.py \
--PLM ./checkpoints \
--output_dir ./results \
--predict_with_generate True \
--generation_max_length 128 \
--generation_num_beams 5 \
--max_input_length 1024
