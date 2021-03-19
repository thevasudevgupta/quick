# quick

`quick` is the simple trainer built on the top of `pytorch` & `deepspeed` for making the your deep learning model training more smoother & faster.

## Supported features

- Single trainer for most of use-cases
- Ease of use & getting rid of boiler-plate code
- Automatic logging with wandb
- Early stopping and automatic saving
- Gradient Accumulation
- Switching between CPU/GPU with no code change
- Multi gpu support with help of `deepspeed`

```python

import quick

class Trainer(quick.Trainer):

    def __init__(self, args, model):
        self.lr = args.lr
        self.setup(model)

        # call this at end only
        super().__init__(args)

    def setup_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
    # define model architecture
    model = .....

    # define dataset
    tr_dataset = .....
    val_dataset = .....

    # load default args
    args = quick.TrainingArgs(lr=1e-4)

    trainer = Trainer(args, model)
    trainer.fit(tr_dataset, val_dataset)
```

## Arguments

```md
enable_deepspeed: bool :: Whether to enable deepspeed engine
local_rank: int :: device on which to put your model, batch

base_dir: str :: root dir for any kind of saving (default = `project_directory`)
load_dir: str :: If specified training stuff and model weights will be loaded from this dir (default = None)
save_epoch_dir: str :: If specified, ckpt will be saved at epoch level if loss decreases

project_name: str :: Project name in wandb (default = None)
wandb_run_name: str :: run name in wandb (default = None)

map_location: torch.device :: argument used in torch.load() while loading model-state-dict (default = torch.device("cuda:0"))
early_stop_n: int :: Enable early stopping by specifying how many epochs to look-up before stopping (default = None)
```

### End Notes

- Currently, this can't be used with models involving multiple optimizers (like GANs).
- Don't forget to send your batch to `self.device`, model will be automatically transferred to `self.device` (you need not care that). `self.device` will be set to GPU `enable_deepspeed=True`.
- Model weights will be in `pytorch_model.bin` file while other training stuff will be in `training.tar`.
