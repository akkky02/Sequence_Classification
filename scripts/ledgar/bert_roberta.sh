#!/bin/bash

# Define an array of model configurations
models=(
    "google-bert/bert-base-uncased,MAdAiLab/bert-base-uncased_ledgar"
    "distilbert/distilbert-base-uncased,MAdAiLab/distilbert-base-uncased_ledgar"
    "FacebookAI/roberta-base,MAdAiLab/roberta-base_ledgar"
    "distilbert/distilroberta-base,MAdAiLab/distilroberta-base_ledgar"
)

# Define common parameters
common_params=(
    --dataset_name "coastalcph/lex_glue"
    --dataset_config_name "ledgar"
    --text_column_name "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --do_train
    --do_eval
    --do_predict
    --max_seq_length 512
    --evaluation_strategy "steps"
    --eval_steps 100
    --save_steps 100
    --load_best_model_at_end
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --eval_accumulation_steps 100
    --optim "adamw_torch"
    --learning_rate 2e-5
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
    accelerate launch --config_file ../../config/default_config.yaml ../run_classification.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_ledgar" \
        --output_dir "../../experiments_checkpoints/MAdAiLab/${model_name//-/_}_ledgar/"
done
