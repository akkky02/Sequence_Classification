#!/bin/bash

# Define an array of dataset configurations
datasets=(
    "MAdAiLab/twitter_disaster,default"
    "MAdAiLab/amazon-attrprompt,default"
    "coastalcph/lex_glue,scotus"
    "ccdv/patent-classification,abstract"
    "coastalcph/lex_glue,ledgar"
)

# Define an array of model configurations
models=(
    # "mistralai/Mistral-7B-v0.1,NA"
    # "meta-llama/Llama-2-7b-hf,NA"
    # "google/gemma-7b,NA"
    # "Qwen/Qwen1.5-7B,NA"
    "mosaicml/mosaic-bert-base-seqlen-2048,NA"
)

# Define common parameters
common_params=(
    --text_column_names "text"
    --label_column_name "label"
    --shuffle_train_dataset
    --trust_remote_code
    --do_train
    --do_eval
    --do_predict
    --bf16
    --gradient_checkpointing
    --max_seq_length 2048
    --evaluation_strategy "steps"
    --eval_steps 50
    --save_steps 50
    --load_best_model_at_end
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
    --logging_steps 10
    --save_total_limit 1
    --save_safetensors False
    --overwrite_output_dir
    --log_level "warning"
    #lora parameters
    # --do_lora
    # --r 128
    # --lora_alpha 256
    # --task_type "SEQ_CLS"
    # --lora_dropout 0.05
)

# Iterate over each dataset and model configuration
for dataset_config in "${datasets[@]}"; do
    IFS=',' read -ra dataset <<< "$dataset_config"
    dataset_name="${dataset[0]}"
    dataset_config_name="${dataset[1]}"

    for model_config in "${models[@]}"; do
        IFS=',' read -ra config <<< "$model_config"
        model_name="${config[0]}"
        hub_model_id="${config[1]}"

        # Run classification with the current model and dataset configuration
        accelerate launch --config_file ../config/deepspeed_config.yaml ./run_classification.py \
            --model_name_or_path "$model_name" \
            --hub_model_id "$hub_model_id" \
            --dataset_name "$dataset_name" \
            --dataset_config_name "$dataset_config_name" \
            "${common_params[@]}" \
            --run_name "${model_name//-/_}_${dataset_name//-/_}_${dataset_config_name}" \
            --output_dir "../experiments_checkpoints/LoRA/${model_name//-/_}_${dataset_name//-/_}_${dataset_config_name}/"
    done
done
