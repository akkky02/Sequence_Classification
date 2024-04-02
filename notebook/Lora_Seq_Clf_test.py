# ## Setup Environment
import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PhiForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "TEST_SEQ_CLASSIFICATION_RUNS"

# ## Load Dataset

raw_datasets = load_dataset("MAdAiLab/twitter_disaster")
# raw_datasets = df.rename_column("label", "labels")


logger = logging.getLogger(__name__)


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


label_list = get_label_list(raw_datasets, split="train")
for split in ["validation", "test"]:
    if split in raw_datasets:
        val_or_test_labels = get_label_list(raw_datasets, split=split)
        diff = set(val_or_test_labels).difference(set(label_list))
        if len(diff) > 0:
            # add the labels that appear in val/test but not in train, throw a warning
            logger.warning(
                f"Labels {diff} in {split} set but not in training set, adding them to the label list"
            )
            label_list += list(diff)
# if label is -1, we throw a warning and remove it from the label list
for label in label_list:
    if label == -1:
        logger.warning("Label -1 found in label list, removing it.")
        label_list.remove(label)

label_list.sort()
num_labels = len(label_list)
if num_labels <= 1:
    raise ValueError("You need more than one label to do classification.")


# ## Load pretrained model and tokenizer


checkpoint = "mistralai/Mistral-7B-v0.1"

config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        num_labels=num_labels,
        finetuning_task="text-classification",
        trust_remote_code=True,
)
config.problem_type = "single_label_classification"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config=config,
        trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
            r=2,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=4,
            task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

label_to_id = {v: i for i, v in enumerate(label_list)}

model.config.label2id = label_to_id
model.config.id2label = {id: label for label, id in label_to_id.items()}

max_seq_length =  4096 


# ## Preprocess Dataset


def preprocess_function(examples):
    # Tokenize the texts
    result = tokenizer(examples["text"], padding="max_length", max_length=max_seq_length, truncation=True)

    return result

raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
)

train_dataset = raw_datasets["train"].select(range(20))
eval_dataset = raw_datasets["validation"].select(range(20))
predict_dataset = raw_datasets["test"].select(range(20))

for index in random.sample(range(len(train_dataset)), 3):
    print(f"Sample {index} of the training set: {train_dataset[index]}.")

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


# ## Compute metric


def compute_metrics(p: EvalPrediction):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
        "f1_micro": f1.compute(predictions=preds, references=p.label_ids, average="micro")["f1"],
    }
    return result


# ## Training Args


training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    do_predict=True,
    bf16=True,
    # fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    max_steps=20,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=5,
    max_grad_norm=1,
    weight_decay=0.1,
    optim="adamw_torch",
    learning_rate=5e-6,
    lr_scheduler_type="linear",
    num_train_epochs=3,
    report_to="wandb",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    save_safetensors=False,
    overwrite_output_dir=True,
    log_level="info",
    output_dir="./test_runs/mistralai/Mistral-7B-v0.1",
)



# ## Initialize Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ## Training, Evaluation and Prediction


def main(training_args):
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = 20
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        max_eval_samples = 20
        metrics = trainer.evaluate(eval_dataset=eval_dataset.select(range(max_eval_samples)))
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(predict_dataset)
        metrics["test_samples"] = len(predict_dataset)
        trainer.log_metrics("test", predictions.metrics)
        trainer.save_metrics("test", predictions.metrics)

if __name__ == "__main__":
    main(training_args)




