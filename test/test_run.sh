#!/bin/bash

# Set the required variables
MODEL_NAME="MAdAiLab/distilbert-base-uncased_twitter"  # Replace with your model name on the Hub
DATASET_NAME="MAdAiLab/twitter_disaster"  # Replace with the name of your dataset
DATASET_CONFIG_NAME="default"  # Replace with the configuration name of your dataset
OUTPUT_DIR="tmp/test_run_only_predict/"  # Replace with the desired output directory path

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the Python script
python test_run.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --do_predict \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --no_train \
    --no_eval