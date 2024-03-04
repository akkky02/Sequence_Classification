#!/bin/bash

python run_classification.py \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name "ccdv/patent-classification"  \
    --dataset_config_name "abstract" \
    --text_column_name "text" \
    --label_column_name "label" \
    --shuffle_train_dataset \
    --do_train \
    --do_eval \
    --do_predict \
    --gradient_checkpointing \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --max_seq_length 512 \
    --load_best_model_at_end \
    --metric_name "f1" \
    --metric_for_best_model "f1" \
    --greater_is_better True \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --optimizer "adamw" \
    --learning_rate 2e-5 \
    --lr_scheduler_type "linear" \
    --num_train_epochs 3 \
    --report_to "wandb" \
    --run_name "distilbert-base-uncased_patent" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --hub_model_id MAdAiLab/distilbert-base-uncased_patent \
    --push_to_hub \
    --output_dir /tmp/MaMAdAiLab/distilbert-base-uncased_patent/