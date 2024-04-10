# SLM vs LLM Sequence Classification

This project aims to perform finetuning for sequence classification tasks and compare the performances of various Encoder, Seq2Seq, and Decoder models. Specifically, we compare models across three categories:

- Encoder models: BERT, RoBERTa, DistilBERT, DistilRoBERTa
- Seq2Seq models: T5, Flan-T5
- Decoder models: Qwen 1.5 billion, Gemma-2B, Phi2, Mistral-7B, and LLAMA 2 7B

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Directly using the `accelerate` command](#directly-using-the-accelerate-command)
  - [Using the provided scripts](#using-the-provided-scripts)
  - [Single GPU Setup](#single-gpu-setup)
  - [Multi GPU Setup (Accelerate Configuration)](#multi-gpu-setup-accelerate-configuration)
- [References](#references)
- [License](#license)

## Project Structure

The project has the following structure:

- `config`: Contains acclerate configuration files for the project.
- `experiments_checkpoints`: Stores checkpoints for different experiments.
- `notebook`: Contains Jupyter notebooks and scripts for data preparation, model experimentation, and testing.
- `scripts`: Contains scripts for running various parts of the project.
- `test`: Contains test runs to ensure the functionality of the code.
- `venv`: Contains a Python virtual environment for managing dependencies.

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment.
4. Activate the virtual environment.
5. Install the dependencies.

```
git clone https://gitlab.com/rohit103/akshat-classification-experiments.git
cd akshat-classification-experiments
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

There are two ways to run the experiments:

### Directly using the ```accelerate``` command

Navigate to the `scripts` directory and run the `accelerate` command with the appropriate configuration file and parameters. For example:

```bash
cd scripts

accelerate launch --config_file ../config/default_config.yaml run_classification.py \
--model_name_or_path "mistralai/Mistral-7B-v0.1" \
--hub_model_id "MAdAiLab/bert-base-uncased_amazon" \
--dataset_name "MAdAiLab/amazon-attrprompt" \
--text_column_name "text" \
--label_column_name "label" \
--shuffle_train_dataset \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 512 \
--evaluation_strategy "steps" \
--eval_steps 50 \
--save_steps 50 \
--load_best_model_at_end \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--optim "adamw_torch" \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--num_train_epochs 3 \
--report_to "wandb" \
--logging_strategy "steps" \
--logging_steps 10 \
--save_total_limit 1 \
--overwrite_output_dir \
# if lora is to be used then add the following parameters as well \
--do_lora \
--r 128 \
--lora_alpha 256 \
--task_type "SEQ_CLS" \
--lora_dropout 0.05
```

Modify the above parameters as needed. 

> **Note:** For lora, the `do_lora` parameter must be set to `True` and the `task_type` parameter must be set to `SEQ_CLS`. The `r`, `lora_alpha`, and `lora_dropout` parameters are specific to the LoRA model and can be modified as needed. Please refer the srcipt [lora.sh](scripts/lora.sh) for more information.

### Using the provided scripts

You can also modify the provided scripts in the [bert_roberta.sh](scripts/amazon/bert_roberta.sh)  to suit your needs and run them directly. For example:

```bash
./bert_roberta.sh
```

The provided scripts iterate over different model configurations and run the `run_classification.py` script with each configuration. The common parameters for all runs are defined in the `common_params` array in the script. You can modify this array to change the parameters for all runs.

The script splits each configuration string into the model name and the hub model ID, and runs the `run_classification.py` script with these values and the common parameters.

### Single GPU Setup

If you're running this project on a single GPU setup, you don't need to use the `accelerate` command. Instead, you can run the `run_classification.py` script directly with Python. For example:

```bash
python run_classification.py --model_name_or_path "google-bert/bert-base-uncased" --hub_model_id "MAdAiLab/bert-base-uncased_amazon" ...
```

### Multi GPU Setup (Accelerate Configuration)

This project uses the `accelerate` library for device placement and distributed training. Two configuration files are provided in the `config` directory:

1. `default_config.yaml`: This is the default configuration for a multi-GPU setup.
2. `deepspeed_config.yaml`: This configuration is for a DeepSpeed Stage 1 multi-GPU setup.

You can use either of these configurations by specifying the `--config_file` parameter when running the `accelerate` command or the provided scripts. For example:

```bash
accelerate launch --config_file ../config/default_config.yaml run_classification.py ...
```
or

```bash
accelerate launch --config_file ../config/deepspeed_config.yaml run_classification.py ...
```

Replace the `...` with the rest of the parameters as needed.

You can create your own configuration file and use it in the same way. To create a new configuration file, you can use the `accelerate config` command and follow the prompts to specify your setup.

For more information on the accelerate library, refer to the [accelerate documentation](https://huggingface.co/docs/accelerate/basic_tutorials/install).


## References
Below are some references that were used to build this project:

- [Hugging Face Transformers github](https://github.com/huggingface/transformers)
- [Hugging Face Accelerate github](https://github.com/huggingface/accelerate)
- [Hugging Face PEFT github](https://github.com/huggingface/peft)
- [Hugging Face Text Classification](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py)
- [Hugging Face Blog Post on Classification](https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md)

## License

Apache License 2.0

