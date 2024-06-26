#!/bin/bash

# Define an array of model configurations
models=(
    "google/gemma-2b,MAdAiLab/gemma_2b_ledgar"
    "microsoft/phi-2,MAdAiLab/phi_2_ledgar"
    "Qwen/Qwen1.5-1.8B,MAdAiLab/Qwen1.5-1.8B_ledgar"
)

# Define common parameters
common_params=(
    --dataset_name "coastalcph/lex_glue"
    --dataset_config_name "ledgar"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --trust_remote_code
    --do_train
    --do_eval
    --do_predict
    --max_seq_length 512
    --gradient_checkpointing
    --evaluation_strategy "steps"
    --eval_steps 100
    --save_steps 100
    --load_best_model_at_end
    --bf16
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --eval_accumulation_steps 100
    --max_grad_norm 1
    --weight_decay 0.1
    --optim "adamw_torch"
    --learning_rate 5e-6
    --lr_scheduler_type "linear"
    --num_train_epochs 3
    --report_to "wandb"
    --logging_strategy "steps"
    --logging_steps 25
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
    accelerate launch --config_file ../../config/deepspeed_config.yaml ../run_classification.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_ledgar" \
        --output_dir "../../experiments_checkpoints/MAdAiLab/${model_name//-/_}_ledgar/"
done