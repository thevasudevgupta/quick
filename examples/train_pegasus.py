"""
RUNNING THIS SCRIPT:
    $ mkdir data
    $ cd data && wget https://huggingface.co/datasets/vasudevgupta/data/resolve/main/train.csv
    $ cd ..
    $ BATCH_SIZE=4 GRAD_ACC_STEPS=1 DATA_FILE_NAME="data/train.csv" python3 train_pegasus.py
    # or
    $ ENABLE_DEEPSPEED=True BATCH_SIZE=8 GRAD_ACC_STEPS=1 deepspeed train_pegasus.py

This will train it with normal torch distributed on multiple gpus
"""

import os

from quick.torch_trainer import DeepSpeedPlugin

import torch
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup

ENABLE_DEEPSPEED = eval(os.environ.pop("ENABLE_DEEPSPEED", "False"))
BATCH_SIZE = eval(os.environ.pop("BATCH_SIZE", "8"))
GRAD_ACC_STEPS = eval(os.environ.pop("GRAD_ACC_STEPS", "4"))
DATA_FILE_NAME = os.environ.pop("DATA_FILE_NAME", "data/train.csv")

MAX_LENGTH = eval(os.environ.pop("MAX_LENGTH", "256"))
MODEL_ID = os.environ.pop("MODEL_ID", "google/pegasus-large")
TEST_SIZE = eval(os.environ.pop("TEST_SIZE", "0.01")) # don't need test data

if ENABLE_DEEPSPEED:
    from quick import DeepSpeedTrainer as TorchTrainer
else:
    from quick import TorchTrainer

def collate_fn(features):
    title = [f['title'] for f in features]
    abstract = [f['abstract'] for f in features]

    inputs = tokenizer(title, return_tensors="pt", max_length=32, truncation=True, padding=True, add_special_tokens=False)
    outputs = tokenizer(abstract, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding=True)
    labels = outputs['input_ids']
#     print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
#     print(tokenizer.convert_ids_to_tokens(labels[0]))
    labels[labels == tokenizer.pad_token_id] = -100
#     print(labels[0])
    # -100 will be replaced by `pad_token_id` inside for `decoder_input_ids`
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        "decoder_attention_mask": outputs['attention_mask'],
    }


class Trainer(TorchTrainer):
    def setup_optimizer(self):
        return Adafactor(self.model.parameters())

#     def setup_scheduler(self):
#         return get_linear_schedule_with_warmup(self.optimizer, 500, 150000)

    def train_on_batch(self, batch, batch_idx):
        batch = {k: batch[k].to(self.device) for k in batch}
        out = self.model(**batch, return_dict=True)
        return out["loss"].mean()

    @torch.no_grad()
    def evaluate_on_batch(self, batch):
        batch = {k: batch[k].to(self.device) for k in batch}
        out = self.model(**batch, return_dict=True)
        return out["loss"].mean()


if __name__ == '__main__':
    from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration
    from quick import TrainingArgs, CONVERT_HF_DATA_TO_TORCH_DATA
    from datasets import load_dataset

    # setting model & tokenizer
    tokenizer = PegasusTokenizerFast.from_pretrained(MODEL_ID)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_ID)

    # setting data
    dataset = load_dataset("csv", data_files=DATA_FILE_NAME)['train']
    print(dataset)
    data = dataset.train_test_split(test_size=TEST_SIZE, seed=42, shuffle=True)
    tr_data, val_data = CONVERT_HF_DATA_TO_TORCH_DATA(data['train']), data['test']

    # setting trainer
    deepspeed_plugin = DeepSpeedPlugin(fp16={"enabled": False}, zero_optimization={"stage": 0})
    args = TrainingArgs(
        output_dir="/workspace/data/pegasus",
        enable_deepspeed=ENABLE_DEEPSPEED,
        deepspeed_plugin=deepspeed_plugin,
        batch_size_per_device=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        max_epochs=3,
    )

    trainer = Trainer(args)
    trainer.fit(model, tr_data, val_data, collate_fn=collate_fn)

    # saving huggingface final model
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final-weights"))
