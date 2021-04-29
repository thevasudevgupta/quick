"""
RUNNING THIS SCRIPT:
    $ mkdir data
    $ cd data && wget https://huggingface.co/datasets/vasudevgupta/data/resolve/main/train.csv
    $ cd ..
    $ BATCH_SIZE=8 GRAD_ACC_STEPS=4 DATA_FILE_NAME="data/train.csv" deepspeed train_pegasus.py

This will train it with normal torch distributed on multiple gpus
"""

import os

from quick.torch_trainer import DeepSpeedPlugin
from quick import DeepSpeedTrainer as TorchTrainer
import torch
from transformers import Adafactor, get_linear_schedule_with_warmup


BATCH_SIZE = eval(os.environ.pop("BATCH_SIZE", "8"))
GRAD_ACC_STEPS = eval(os.environ.pop("GRAD_ACC_STEPS", "4"))
DATA_FILE_NAME = os.environ.pop("DATA_FILE_NAME", "data/train.csv")

MAX_LENGTH = eval(os.environ.pop("MAX_LENGTH", "256"))
MODEL_ID = os.environ.pop("MODEL_ID", "google/pegasus-large")
TEST_SIZE = eval(os.environ.pop("TEST_SIZE", "0.01")) # don't need test data


def get_dataloader(dataset, tokenizer, shuffle=False):
    def collate_fn(features):
        title = [f['title'] for f in features]
        abstract = [f['abstract'] for f in features]
        batch = tokenizer(title, abstract, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding=True)
        labels = batch['input_ids'].clone()
        # ingoring padding index in loss
        labels[labels == tokenizer.pad_token_id] = -100
        # -100 will be replaced by `pad_token_id` inside for `decoder_input_ids`
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=shuffle, pin_memory=True, num_workers=2)

# TODO fix dataloader
class Trainer(TorchTrainer):
    def setup_optimizer(self):
        return Adafactor(self.model.parameters(), lr=self.args.lr)

    def setup_scheduler(self):
        return get_linear_schedule_with_warmup(self.optimizer, WARMUP, )

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
    from quick import TrainingArgs
    from datasets import load_dataset

    # setting model & tokenizer
    tokenizer = PegasusTokenizerFast.from_pretrained(MODEL_ID)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_ID)

    # setting data
    dataset = load_dataset("csv", data_files=DATA_FILE_NAME)['train']
    data = dataset.train_test_split(test_size=TEST_SIZE, seed=42, shuffle=True)
    tr_data, val_data = data['train'], data['test']
    print(tr_data, val_data)
    tr_data = get_dataloader(tr_data, tokenizer=tokenizer, shuffle=True)
    val_data = get_dataloader(val_data, tokenizer=tokenizer)

    # setting trainer
    deepspeed_plugin = DeepSpeedPlugin(fp16={"enabled": False}, zero_optimization={"stage": 0})
    args = TrainingArgs(
        enable_deepspeed=True,
        deepspeed_plugin=deepspeed_plugin,
        lr=5.e-5,
        batch_size_per_device=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
    )
    trainer = Trainer(args)

    # training model
    trainer.setup(model)
    trainer.fit(tr_data, val_data)

    # saving huggingface final model
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final-weights"))
