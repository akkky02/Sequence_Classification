#!/bin/bash

# Define an array of model configurations
models=(
    # "distilbert/distilroberta-base,MAdAiLab/distilroberta-base_twitter"
    # "google/gemma-2b,MAdAiLab/distilbert-base-uncased_twitter"
    # "openai-community/gpt2,MAdAiLab/gpt2_twitter"
    # "microsoft/phi-2,MAdAiLab/microsoft_phi_2_twitter"
    # "facebook/opt-1.3b,MAdAiLab/facebook_opt_1.3b_twitter"
    # "Qwen/Qwen1.5-1.8B,MAdAiLab/Qwen1.5-1.8B_twitter"
    # "mistralai/Mistral-7B-v0.1,MAdAiLab/Mistral-7B-v0.1_twitter"
    # "meta-llama/Llama-2-7b-hf,MAdAiLab/Llama-2-7b-hf_twitter"
    # "allenai/OLMo-1B,MAdAiLab/OLMo_1B_twitter"
    "mosaicml/mosaic-bert-base-seqlen-2048,NA"
)

# Define common parameters
common_params=(
    --dataset_name "MAdAiLab/twitter_disaster"
    --text_column_names "text"
    --label_column_name "label"
    --use_fast_tokenizer
    --max_seq_length 2048
    --shuffle_train_dataset
    --trust_remote_code
    --do_train
    --do_eval
    --do_predict
    --bf16
    # --gradient_accumulation_steps 1
    --max_steps 200
    --evaluation_strategy "steps"
    --eval_steps 50
    --save_steps 50
    --load_best_model_at_end
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --eval_accumulation_steps 5
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
    # --r 2
    # --lora_alpha 2
    # --task_type "SEQ_CLS"
)

# Iterate over each model configuration
for model_config in "${models[@]}"; do
    IFS=',' read -ra config <<< "$model_config"
    model_name="${config[0]}"
    hub_model_id="${config[1]}"

    # Run classification with the current model configuration
    accelerate launch --config_file ../config/deepspeed_config.yaml test_run.py \
        --model_name_or_path "$model_name" \
        --hub_model_id "$hub_model_id" \
        "${common_params[@]}" \
        --run_name "${model_name//-/_}_twitter" \
        --output_dir "./test_runs/MAdAiLab/${model_name//-/_}_twitter/"
done
