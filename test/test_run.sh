#!/bin/bash

# Define an array of model configurations
models=(
    # "google/gemma-2b,MAdAiLab/distilbert-base-uncased_twitter"
    # "openai-community/gpt2,MAdAiLab/gpt2_twitter"
    # "microsoft/phi-2,MAdAiLab/microsoft_phi_2_twitter"
    "facebook/opt-1.3b,MAdAiLab/facebook_opt_1.3b_twitter"
    # "Qwen/Qwen1.5-1.8B,MAdAiLab/Qwen1.5-1.8B_twitter"
)

# Define common parameters
common_params=(
    --dataset_name "MAdAiLab/twitter_disaster"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --max_train_samples 100
    --max_eval_samples 25
    --max_predict_samples 25
    --trust_remote_code
    --do_train
    --do_eval
    --do_predict
    --gradient_checkpointing
    --evaluation_strategy "steps"
    --eval_steps 10
    --load_best_model_at_end
    --metric_name "f1"
    --metric_for_best_model "f1"
    --greater_is_better True
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 8
    --eval_accumulation_steps 20
    --optim "adafactor"
    --learning_rate 2e-5
    --lr_scheduler_type "linear"
    --num_train_epochs 3
    --report_to "wandb"
    --logging_strategy "steps"
    --logging_steps 10
    --overwrite_output_dir
)

# Iterate over each model configuration
for model_config in "${models[@]}"; do
    IFS=',' read -ra config <<< "$model_config"
    model_name="${config[0]}"
    hub_model_id="${config[1]}"

    # Run classification with the current model configuration
    python test_run.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_twitter" \
        --output_dir "./test_runs/MAdAiLab/${model_name//-/_}_twitter/"
done
