
import quick
import torch

import os
import unittest

class DummyModel(torch.nn.Module):

    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')

    def forward(self, batch):
        return batch

    def save_pretrained(self, save_dir):
        path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(path, self.state_dict())


class Trainer(quick.Trainer):

    def __init__(self, args, model):
        self.lr = args.lr
        self.setup(model)

    def setup_optimizer(self):
        return torch.optim.Adam(self.model.parameters, lr=self.lr)

    def setup_scheduler(self):
        return

    def train_batch(self, batch, batch_idx):

        batch = {k: batch[k].to(self.device) for k in batch}
        outputs = self.model(**batch)
        loss = outputs["loss"].mean()
        return loss

    @torch.no_grad()
    def validate_batch(self, batch, batch_idx):

        batch = {k: batch[k].to(self.device) for k in batch}
        outputs = self.model(**batch)
        loss = outputs["loss"].mean()
        return loss

    def training_epoch_end(self, epoch, tr_metric, val_metric):
        self.model.save_pretrained("dummy")


class QuickTrainerTest(unittest.TestCase):

    def test_trainer(self):

        model = DummyModel()

        args = quick.TrainingArgs()
        trainer = Trainer(args, model)

        trainer.fit(tr_dataset, val_dataset)
