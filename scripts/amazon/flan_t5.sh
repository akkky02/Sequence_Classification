#!/bin/bash

# Define an array of model configurations
models=(
    "google/flan-t5-small,MAdAiLab/flan-t5-small_amazon"
    "google/flan-t5-base,MAdAiLab/flan-t5-base_amazon"
    "google-t5/t5-small,MAdAiLab/t5-small_amazon"
    "google-t5/t5-base,MAdAiLab/t5-base_amazon"
)

# Define common parameters
common_params=(
    --dataset_name "MAdAiLab/amazon-attrprompt"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --do_train
    --do_eval
    --do_predict
    --evaluation_strategy "steps"
    --eval_steps 50
    --save_steps 50
    --load_best_model_at_end
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16
    --optim "adamw_torch"
    --learning_rate 5e-4
    --lr_scheduler_type "linear"
    --num_train_epochs 3
    --report_to "wandb"
    --logging_strategy "steps"
    --logging_steps 10
    --save_total_limit 1
    --overwrite_output_dir
    --log_level "warning"
)

# Iterate over each model configuration
for model_config in "${models[@]}"; do
    IFS=',' read -ra config <<< "$model_config"
    model_name="${config[0]}"
    hub_model_id="${config[1]}"

    # Run classification with the current model configuration
    accelerate launch --config_file ../../config/default_config.yaml ../run_classification.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_amazon" \
        --output_dir "../../experiments_checkpoints/MAdAiLab/${model_name//-/_}_amazon/"
done
