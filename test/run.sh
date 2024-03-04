#!/bin/bash

python ../scripts/run_classification.py \
    --model_name_or_path "MAdAiLab/bert-base-uncased_patent" \
    --dataset_name "ccdv/patent-classification"  \
    --dataset_config_name "abstract" \
    --text_column_name "text" \
    --label_column_name "label" \
    --do_predict \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --max_seq_length 512 \
    --metric_name "f1" \
    --metric_for_best_model "f1" \
    --greater_is_better True \
    --per_device_eval_batch_size 64 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --hub_model_id MAdAiLab/flan-t5-small_patent \
    --push_to_hub \
    --output_dir /tmp/MAdAiLab/flan-t5-small_patent/