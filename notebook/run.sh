#!/bin/bash

accelerate launch --config_file ../config/deepspeed_config.yaml Lora_Seq_Clf_test.py