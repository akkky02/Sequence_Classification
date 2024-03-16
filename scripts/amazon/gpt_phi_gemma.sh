#!/bin/bash

# Define an array of model configurations
models=(
    "google/gemma-2b,MAdAiLab/gemma_2b_amazon"
    "microsoft/phi-2,MAdAiLab/phi_2_amazon"
    "Qwen/Qwen1.5-1.8B,MAdAiLab/Qwen1.5-1.8B_amazon"
)

# Define common parameters
common_params=(
    --dataset_name "MAdAiLab/amazon-attrprompt"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --trust_remote_code
    --do_train
    --do_eval
    --do_predict
    --evaluation_strategy "steps"
    --eval_steps 50
    --save_steps 50
    --load_best_model_at_end
    --bf16
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16
    --eval_accumulation_steps 50
    --max_grad_norm 1
    --weight_decay 0.1
    --optim "adamw_torch"
    --learning_rate 5e-6
    # --warmup_steps 200
    --lr_scheduler_type "linear"
    --num_train_epochs 1
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
    accelerate launch --config_file ../../config/deepspeed_config.yaml ../run_classification.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_amazon" \
        --output_dir "../../experiments_checkpoints/MAdAiLab/${model_name//-/_}_amazon/"
done


# # Define an array of model configurations
# models=(
#     "google/gemma-2b,MAdAiLab/gemma_2b_patent"
#     "microsoft/phi-2,MAdAiLab/phi_2_patent"
#     "Qwen/Qwen1.5-1.8B,MAdAiLab/Qwen1.5-1.8B_patent"
# )

# # Define common parameters
# common_params=(
#     --dataset_name "ccdv/patent-classification"
#     --dataset_config_name "abstract"
#     --text_column_name "text"
#     --label_column_name "label"
#     --shuffle_train_dataset
#     --trust_remote_code
#     --do_train
#     --do_eval
#     --do_predict
#     --evaluation_strategy "steps"
#     --eval_steps 50
#     --load_best_model_at_end
#     --bf16
#     --metric_name "f1"
#     --metric_for_best_model "f1"
#     --greater_is_better True
#     --per_device_train_batch_size 8
#     --per_device_eval_batch_size 8
#     --eval_accumulation_steps 50
#     --max_grad_norm 1
#     --weight_decay 0.1
#     --optim "adafactor"
#     --learning_rate 5e-6
#     # --warmup_steps 200
#     --lr_scheduler_type "linear"
#     --num_train_epochs 3
#     --report_to "wandb"
#     --logging_strategy "steps"
#     --logging_steps 10
#     --save_total_limit 3
#     --overwrite_output_dir
# )

# # Iterate over each model configuration
# for model_config in "${models[@]}"; do
#     IFS=',' read -ra config <<< "$model_config"
#     model_name="${config[0]}"
#     hub_model_id="${config[1]}"

#     # Run classification with the current model configuration
#     python ../run_classification.py \
#         --model_name_or_path "$model_name" \
#         --hub_model_id "$hub_model_id" \
#         "${common_params[@]}" \
#         --run_name "${model_name//-/_}_patent" \
#         --output_dir "./experiments/MAdAiLab/${model_name//-/_}_patent/"
# done