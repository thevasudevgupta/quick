
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from datasets import load_dataset

import quick
from dataclasses import dataclass


@dataclass
class TrainingArgs(quick.TrainingArgs):

    lr: float = 1.e-5
    batch_size: int = 1
    max_epochs: int = 10
    accumulation_steps: int = 8

    num_workers: int = 2
    max_length: int = 512
    max_target_length: int = 32
    process_on_fly: bool = False
    seed: int = 42
    n_augment: int = 2

    file_path: str = "data/dev_data_article.csv"
    pretrained_model_id: str = "facebook/mbart-large-cc25"
    pretrained_tokenizer_id: str = "facebook/mbart-large-cc25"
    weights_dir: str = "test-quick"

    base_dir: str = "test-quick"
    wandb_run_name: str = "test-quick"
    project_name: str = "interiit-mbart"


class DataLoader(object):

    def __init__(self, tokenizer, args):

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        self.file_path = args.file_path

        self.tokenizer = tokenizer
        self.sep_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id)

    def setup(self, ):

        data = load_dataset("csv", data_files=self.file_path)["train"]
        data = data.map(lambda x: {"CleanedHeadline": x["Headline"]})
        data = data.filter(lambda x: x["article_length"] > 32 and x["summary_length"] > 1)

        data = data.filter(lambda x: type(x["CleanedHeadline"]) == str and type(x["CleanedText"]) == str)
        print("Dataset", data)

        data = data.train_test_split(test_size=600, shuffle=True, seed=42)
        tr_dataset = data["train"].map(lambda x: {"split": "TRAIN"})
        val_dataset = data["test"].map(lambda x: {"split": "VALIDATION"})

        return tr_dataset, val_dataset

    def train_dataloader(self, tr_dataset):
        return torch.utils.data.DataLoader(
            tr_dataset,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self, val_dataset):
        return torch.utils.data.DataLoader(
            val_dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def collate_fn(self, features):
        article = [f["CleanedText"] for f in features]
        summary = [f["CleanedHeadline"] for f in features]
        # src_lang will be dummy
        features = self.tokenizer.prepare_seq2seq_batch(
            src_texts=article, src_lang="hi_IN", tgt_lang="en_XX", tgt_texts=summary, truncation=True, 
            max_length=self.max_length, max_target_length=self.max_target_length, return_tensors="pt"
        )
        return features


class Trainer(quick.Trainer):

    def __init__(self, model, args):
        self.lr = args.lr

        self.setup(model)
        super().__init__(args)

    def setup_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_on_batch(self, batch, batch_idx):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    @torch.no_grad()
    def evaluate_on_batch(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    def training_epoch_end(self, epoch, losses):
        # saving state_dict at epoch level
        if self.args.weights_dir:
            self.model.save_pretrained(os.path.join(self.args.base_dir, self.args.weights_dir+f"-e{epoch}"))


if __name__ == '__main__':

    args = TrainingArgs()

    model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_id)
    tokenizer = MBartTokenizer.from_pretrained(args.pretrained_tokenizer_id)

    dl = DataLoader(tokenizer, args)
    tr_dataset, val_dataset = dl.setup()
    print(tr_dataset, val_dataset)

    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)
