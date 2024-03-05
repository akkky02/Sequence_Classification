#!/bin/bash

# Set the required variables
MODEL_NAME="MAdAiLab/distilbert-base-uncased_twitter"  # Replace with your model name on the Hub
DATASET_NAME="MAdAiLab/twitter_disaster"  # Replace with the name of your dataset
OUTPUT_DIR="/tmp/test_run_only_predict/"  # Replace with the desired output directory path

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the Python script
python test_run.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --text_column_name "text" \
    --label_column_name "label" \
    --metric_name "f1" \
    --metric_for_best_model "f1" \
    --per_device_eval_batch_size 64 \
    --do_eval \
    --do_predict \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
