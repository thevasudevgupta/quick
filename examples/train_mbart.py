
import os

import torch
from datasets import load_dataset

ENABLE_DEEPSPEED = eval(os.environ.pop("ENABLE_DEEPSPEED", "False"))
FILE_PATH = os.environ.pop("FILE_PATH", "examples/clean_article.csv")
BATCH_SIZE = os.environ.pop("BATCH_SIZE", 1)
PRETRAINED_ID = os.environ.pop("PRETRAINED_ID", "facebook/mbart-large-cc25")

if ENABLE_DEEPSPEED:
    from quick import DeepSpeedTrainer as TorchTrainer
else:
    from quick import TorchTrainer

class DataLoader(object):
    def __init__(self, tokenizer):

        self.max_length = 256
        self.max_target_length = 32

        self.file_path = FILE_PATH

        self.tokenizer = tokenizer
        self.sep_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id)

    def setup(self):

        data = load_dataset("csv", data_files=self.file_path)["train"]
        data = data.filter(lambda x: len(x["Text"]) > 32*4 and len(x["Headline"]) > 1*4)
        data = data.filter(lambda x: type(x["Headline"]) == str and type(x["Text"]) == str)
        print(data)

        data = data.train_test_split(test_size=600, shuffle=True, seed=42)
        return data["train"], data["test"]

    def train_dataloader(self, tr_dataset):
        return torch.utils.data.DataLoader(
            tr_dataset,
            pin_memory=True,
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=self.collate_fn,
            num_workers=2
        )

    def val_dataloader(self, val_dataset):
        return torch.utils.data.DataLoader(
            val_dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=BATCH_SIZE,
            collate_fn=self.collate_fn,
            num_workers=2
        )

    def collate_fn(self, features):
        article = [f["Text"] for f in features]
        summary = [f["Headline"] for f in features]
        # src_lang will be dummy
        features = self.tokenizer.prepare_seq2seq_batch(
            src_texts=article, src_lang="hi_IN", tgt_lang="en_XX", tgt_texts=summary, truncation=True, 
            max_length=self.max_length, max_target_length=self.max_target_length, return_tensors="pt"
        )
        return features


class Trainer(TorchTrainer):
    def __init__(self, args):
        super().__init__(args)

    def setup_optimizer(self):
        if ENABLE_DEEPSPEED:
            return super().setup_optimizer()
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def setup_scheduler(self):
        if ENABLE_DEEPSPEED:
            return super().setup_scheduler()

    def train_on_batch(self, batch, batch_idx):
        batch = {k: batch[k].to(self.device) for k in batch}
        out = self.model(**batch, return_dict=True)
        return out["loss"].mean()

    @torch.no_grad()
    def evaluate_on_batch(self, batch):
        batch = {k: batch[k].to(self.device) for k in batch}
        out = self.model(**batch, return_dict=True)
        return out["loss"].mean()


if __name__ == "__main__":

    from quick import TrainingArgs
    from transformers import MBartForConditionalGeneration, MBartTokenizer
    args = TrainingArgs(enable_deepspeed=ENABLE_DEEPSPEED, batch_size=BATCH_SIZE)
    print(args)

    model = MBartForConditionalGeneration.from_pretrained(PRETRAINED_ID)
    tokenizer = MBartTokenizer.from_pretrained(PRETRAINED_ID)

    dl = DataLoader(tokenizer)
    tr_dataset, val_dataset = dl.setup()
    print(tr_dataset, val_dataset)

    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    trainer = Trainer(args)
    trainer.setup(model)
    trainer.fit(tr_dataset, val_dataset)
