# quick

`quick` is built on the top of `pytorch` & `deepspeed` for making the my deep learning model training more smoother & faster.

## Supported features

- Single trainer for most of use-cases
- Ease of use & getting rid of boiler-plate code
- Automatic logging with wandb
- Early stopping and automatic saving
- Gradient Accumulation
- Distributed Training support with help of `deepspeed`

```python
import os

ENABLE_DEEPSPEED = eval(os.environ.pop("ENABLE_DEEPSPEED", "False"))

if ENABLE_DEEPSPEED:
    from quick import DeepSpeedTrainer as TorchTrainer
else:
    from quick import TorchTrainer


class Trainer(Trainer):
    def __init__(self, args):
        super().__init__()

    def setup_optimizer(self):
        # update this if you want
        return super().setup_optimizer()

    def setup_scheduler(self):
        # update this if you want
        return super().setup_scheduler()

    def training_step(self, batch, batch_idx):
        # This method should look something like this
        batch = batch.to(self.device)
        out = self.model(batch)
        loss = out.mean()
        return loss

    def validation_step(self, batch):
        # This method should look something like this
        batch = batch.to(self.device)
        with torch.no_grad():
            out = self.model(batch)
            loss = out.mean()
        return loss


if __name__ == "__main__":
    from quick import TrainingArgs

    # define model architecture
    model = .....

    # define dataset
    tr_dataset = .....
    val_dataset = .....

    # load default args
    args = TrainingArgs(enable_deepspeed=ENABLE_DEEPSPEED)

    trainer = Trainer(args)
    trainer.setup(model)
    trainer.fit(tr_dataset, val_dataset)
```

## Arguments

```md
output_dir: str :: root dir for any kind of saving (default = `Quick-project`)
load_dir: str :: If specified training stuff and model weights will be loaded from this dir (default = None)
save_strategy: str :: If set to "epoch", ckpt will be saved at epoch level (default="epoch")

project_name: str :: Project name in wandb (default = None)
wandb_run_name: str :: run name in wandb (default = None)

map_location: torch.device :: argument used in torch.load() while loading model-state-dict (default = torch.device("cuda:0"))
early_stop_n: int :: Enable early stopping by specifying how many epochs to look-up before stopping (default = None)

enable_deepspeed: bool :: Whether to enable deepspeed engine
```

### End Notes

- Currently, this can't be used with models involving multiple optimizers (like GANs).
- Don't forget to send your batch to `self.device`, model will be automatically transferred to appropriate device (you need not care that).
