#!/bin/bash

# Define an array of model configurations
models=(
    "google-bert/bert-base-uncased,MAdAiLab/bert-base-uncased_patent"
    "distilbert/distilbert-base-uncased,MAdAiLab/distilbert-base-uncased_patent"
    "FacebookAI/roberta-base,MAdAiLab/roberta-base_patent"
    "distilbert/distilroberta-base,MAdAiLab/distilroberta-base_patent"
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
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --optim "adamw_torch"
    --learning_rate 2e-5
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
