# ENABLE_DEEPSPEED=False BATCH_SIZE=8 GRAD_ACC_STEPS=4 DATA_FILE_NAME="data/train.csv" python3 train_gpt.py
# ENABLE_DEEPSPEED=True BATCH_SIZE=16 GRAD_ACC_STEPS=2 DATA_FILE_NAME="data/train.csv" deepspeed train_gpt.py

import os
from quick.torch_trainer import DeepSpeedPlugin
import torch

ENABLE_DEEPSPEED = eval(os.environ.pop("ENABLE_DEEPSPEED", "False"))
BATCH_SIZE = eval(os.environ.pop("BATCH_SIZE", "8"))
GRAD_ACC_STEPS = eval(os.environ.pop("GRAD_ACC_STEPS", "4"))
MAX_LENGTH = eval(os.environ.pop("MAX_LENGTH", "512"))
DATA_FILE_NAME = os.environ.pop("DATA_FILE_NAME", "data/train.csv")
MODEL_ID = os.environ.pop("MODEL_ID", "distilgpt2")
TEST_SIZE = eval(os.environ.pop("TEST_SIZE", "0.06"))

if ENABLE_DEEPSPEED:
    from quick import DeepSpeedTrainer as TorchTrainer
else:
    from quick import TorchTrainer


def get_dataloader(dataset, tokenizer, shuffle=False):
    def collate_fn(features):
        title = [f['title'] for f in features]
        abstract = [f['abstract'] for f in features]
        batch = tokenizer(title, abstract, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding=True)
        labels = batch['input_ids'].clone()
        # ingoring padding index in loss
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=shuffle, pin_memory=True, num_workers=2)


class Trainer(TorchTrainer):
    def train_on_batch(self, batch, batch_idx):
        batch = {k: batch[k].to(self.device) for k in batch}
        return self.model(**batch)["loss"]
    
    @torch.no_grad()
    def evaluate_on_batch(self, batch):
        batch = {k: batch[k].to(self.device) for k in batch}
        return self.model(**batch)["loss"]


if __name__ == '__main__':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    from quick import TrainingArgs
    from datasets import load_dataset

    # setting model & tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

    # setting data
    dataset = load_dataset("csv", data_files=DATA_FILE_NAME)['train']
    data = dataset.train_test_split(test_size=TEST_SIZE, seed=42, shuffle=True)
    tr_data, val_data = data['train'], data['test']
    print(tr_data, val_data)
    tr_data = get_dataloader(tr_data, tokenizer=tokenizer, shuffle=True)
    val_data = get_dataloader(val_data, tokenizer=tokenizer)

    # setting trainer
    deepspeed_plugin = DeepSpeedPlugin(fp16={"enabled": True}, zero_optimization={"stage": 2, "cpu_offload": True})
    args = TrainingArgs(
        enable_deepspeed=ENABLE_DEEPSPEED,
        deepspeed_plugin=deepspeed_plugin,
        lr=5.e-5,
        batch_size_per_device=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
    )
    trainer = Trainer(args)

    # training model
    trainer.setup(model)
    trainer.fit(tr_data, val_data)
