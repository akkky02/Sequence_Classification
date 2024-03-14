#!/bin/bash

# Define an array of model configurations
models=(
    "google/flan-t5-base,MAdAiLab/flan-t5-base_patent"
    "google/flan-t5-small,MAdAiLab/flan-t5-small_patent"
    "google-t5/t5-small,MAdAiLab/t5-small_patent"
    "google-t5/t5-base,MAdAiLab/t5-base_patent"
)

# Define common parameters
common_params=(
    --dataset_name "ccdv/patent-classification"
    --dataset_config_name "abstract"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --do_train
    --do_eval
    --do_predict
    --evaluation_strategy "steps"
    --eval_steps 50
    --max_seq_length 512
    --load_best_model_at_end
    --metric_name "f1"
    --metric_for_best_model "f1"
    --greater_is_better True
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16
    --eval_accumulation_steps 100
    --optim "adamw_torch"
    --learning_rate 5e-4
    --lr_scheduler_type "linear"
    --num_train_epochs 3
    --report_to "wandb"
    --logging_strategy "steps"
    --logging_steps 10
    --save_total_limit 3
    --overwrite_output_dir
)

# Iterate over each model configuration
for model_config in "${models[@]}"; do
    IFS=',' read -ra config <<< "$model_config"
    model_name="${config[0]}"
    hub_model_id="${config[1]}"

    # Run classification with the current model configuration
    python ../run_classification.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_patent" \
        --output_dir "./experiments/MAdAiLab/${model_name//-/_}_patent/"
done
