# quick

`quick` is built on the top of [`pytorch`](https://github.com/pytorch/pytorch) & [`deepspeed`](https://github.com/microsoft/DeepSpeed) for making my deep learning model training more smoother & faster.

```shell
# install torch compatible with CUDA 11.0 (on Colab & Qblocks)
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# installing quick
pip3 install git+https://github.com/vasudevgupta7/quick@main
```

**How to use?**

```python
>>> import os
>>> ENABLE_DEEPSPEED = eval(os.environ.pop("ENABLE_DEEPSPEED", "False"))

>>> if ENABLE_DEEPSPEED:
...     from quick import DeepSpeedTrainer as TorchTrainer
... else:
...     from quick import TorchTrainer

>>> class Trainer(TorchTrainer):
...     def train_on_batch(self, batch, batch_idx):
...         batch = {k: batch[k].to(self.device) for k in batch}
...         out = self.model(**batch, return_dict=True)
...         return out["loss"].mean()

...     @torch.no_grad()
...     def evaluate_on_batch(self, batch):
...         batch = {k: batch[k].to(self.device) for k in batch}
...         out = self.model(**batch, return_dict=True)
...         return out["loss"].mean()

>>> from quick import TrainingArgs
>>> args = TrainingArgs(enable_deepspeed=ENABLE_DEEPSPEED)
>>> trainer = Trainer(args)
>>> trainer.setup(model)

>>> trainer.fit(tr_dataset, val_dataset)
```

```shell
# without deespeed
ENABLE_DEEPSPEED=False python3 train.py

# with deepspeed
ENABLE_DEEPSPEED=True deepspeed --num_gpus 2 train.py
```

**How can I change TrainingArgs & DeepSpeed config**

```python
>>> from quick import TrainingArgs, DeepSpeedPlugin

>>> args = TrainingArgs(
...     batch_size=8,
...     lr=5e-5,
...     gradient_accumulation_steps=1,
...     max_epochs=3,
...     output_dir="Quick-project", # everything related to the experiment will be saved here
...     save_strategy="epoch", # None
...     project_name="Quick-project",
...     wandb_run_name=None,
...     # deepspeed args
...     enable_deepspeed=False,
...     deepspeed_plugin=DeepSpeedPlugin(enable_deepspeed=True, fp16=(("enabled", False))),
...     )
```

**End Notes**

- Currently, this can't be used with models involving multiple optimizers (like GANs).
- Don't forget to send your batch to `self.device`, model will be automatically transferred (you need not worry about that).
