# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'


import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field

import wandb
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


try:
    import deepspeed
    IS_DEEPSPEED_AVAILABLE = True
except ImportError:
    logger.warning("DeepSpeed is not available => run `pip3 install deepspeed` if you want to use DeepSpeed")
    IS_DEEPSPEED_AVAILABLE = False

"""
USAGE:

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

    >>> from quick import TrainingArgs, DeepSpeedPlugin
    >>> deepspeed_plugin = DeepSpeedPlugin(zero_optimization={"stage": 0})
    >>> args = TrainingArgs(enable_deepspeed=ENABLE_DEEPSPEED, deepspeed_plugin=deepspeed_plugin)
    >>> trainer = Trainer(args)
    >>> trainer.setup(model)

    >>> trainer.fit(tr_dataset, val_dataset)
"""

@dataclass
class DeepSpeedPlugin:
    # following arguments will hold only if `enable_deepspeed=True`
    # this will take all value you would have specified in `ds_config.json`

    local_rank: int  = 0
    train_batch_size: int = None
    gradient_accumulation_steps: int = None

    steps_per_print: int = 10000
    wall_clock_breakdown: bool = False

    fp16: dict = field(default_factory=lambda: {"enabled": True})
    zero_optimization: dict = field(default_factory=lambda: {"stage": 2, "cpu_offload": False})


@dataclass
class TrainingArgs:

    batch_size_per_device: int = 8
    num_workers: int = 2
    lr: float = 5e-5

    gradient_accumulation_steps: int = 1
    precision: str = "float32"

    max_epochs: int = 3
    output_dir: str = "Quick-project" # everything related to the experiment will be saved here
    save_strategy: str = "epoch" # None

    project_name: str = "Quick-project"
    wandb_run_name: str = None

    early_stop_n: int = None
    epoch_saving_n: int = None

    # deepspeed args
    enable_deepspeed: bool = False
    deepspeed_plugin: DeepSpeedPlugin = DeepSpeedPlugin()

    def __post_init__(self):

        assert self.save_strategy in [None, "epoch"], f"save_strategy can either be None / epoch; but you specified {self.save_strategy}"
        assert isinstance(self.deepspeed_plugin, DeepSpeedPlugin), "deepspeed_plugin must be instance of DeepSpeedPlugin"
        assert self.precision in ['float32', 'mixed16'], f"precision can either be float32 / mixed16; but you specified {self.precision}"
        if self.enable_deepspeed:
            assert torch.cuda.is_available(), "GPU is not available => Can't enable DeepSpeed"
            assert IS_DEEPSPEED_AVAILABLE, "DeepSpeed not installed => Run `pip3 install deepspeed`"

        if not torch.cuda.is_available():
            logger.warning("[Quick WARNING] CUDA is not available => Training will happen on CPU")

        self.output_dir = self._setup_dir(self.output_dir)

        if self.precision == "mixed16":
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            raise NotImplementedError # TODO: need to fix some bugs

        if self.save_strategy is None:
            logger.warning("[Quick WARNING] You are not saving anything")

        # this will be overwritten in DeepSpeedTrainer.setup() / TorchTrainer.setup()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        num_gpus = torch.cuda.device_count()

        train_batch_size = self.batch_size_per_device * self.gradient_accumulation_steps * num_gpus
        self.train_batch_size = train_batch_size

        if self.enable_deepspeed:
            # setting up deepspeed plugin
            self.deepspeed_plugin = replace(
                self.deepspeed_plugin,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps
            )

            self.precision = None
            self.batch_size_per_device = None
            self.gradient_accumulation_steps = None
            self.device = None

    @staticmethod
    def _setup_dir(output_dir: str):
        output_dir = "." if not isinstance(output_dir, str) else output_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        return output_dir


class TrainerSetup(object):
    @staticmethod
    def assert_epoch_saving(val_metric: list, n: int = 3, mode: str = "min"):
        """
        Allows saving if loss decreases / accuracy increases
        n = 'min' corresponds to val_metric being loss-metric
        n = 'max' corresponds to val_metric being accuracy-metric

        Note:
            val_metric should be having current value of loss/accuracy
        """
        status = False
        if len(val_metric) < n+1:
            return True

        current_val = val_metric[-1]
        compr = val_metric[-n-2:-2]
        if mode == "min":
            compr = np.min(compr)
            if current_val < compr:
                status = True
        elif mode == "max":
            compr = np.max(compr)
            if current_val > compr:
                status = True
        else:
            raise ValueError("mode can be only either max or min")
        return status

    @staticmethod
    def assert_early_stop(val_metric: list, n: int = None, mode="min"):
        """
        If n is specified, then only early stopping will be enabled

        n = 'min' corresponds to val_metric being loss-metric
        n = 'max' corresponds to val_metric being accuracy-metric
        """
        assert(mode in ["max", "min"])

        stop_status = False
        if n is None:
            return stop_status

        compr = np.max(val_metric[-n:]) if mode == "min" else np.min(val_metric[-n:])
        if compr == val_metric[-1]:
            stop_status = True

        return stop_status

    def stop_early(self, val_metric: list, n: int = None, mode="min"):
        status = self.assert_early_stop(val_metric, n, mode)
        if status:
            raise KeyboardInterrupt("Model training stopped due to early-stopping")

    @staticmethod
    def setup_wandb(args: dict):
        run = wandb.init(project=args["project_name"], name=args["wandb_run_name"], config=args["wandb_config"], dir=args["wandb_dir"])
        return run

    @staticmethod
    def display_metrics(epochs: list, tr_metrics: list, val_metrics: list):
        # round factors should be 3 for proper layout

        results = """
                    |--------------------------------------------|
                        epoch   |   tr_metric   |   val_metric   
                    |--------------------------------------------|"""

        for e, tr, val in zip(range(epochs), tr_metrics, val_metrics):
            res = """
                          {}     |     {}     |      {}     
                    |--------------------------------------------|""".format(
                        np.round(e, 3), np.round(tr, 3), np.round(val, 3)
                        )
            results += res
        print(results)

    @staticmethod
    def model_summary(model: torch.nn.Module):

        num = np.sum([p.nelement() for p in model.parameters()])

        s = {"Network": num}
        for n, layer in model.named_children():
            num = np.sum([p.nelement() for p in layer.parameters()])
            s.update({n: num})

        print("Layers | Parameters")
        for l, p in s.items():
            print("{} | {}".format(l, p))


class TorchTrainer(ABC, TrainerSetup):
    def setup_optimizer(self):
        """This method can be implemented in the class inherited from this class"""
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def setup_scheduler(self):
        """This method can be implemented in the class inherited from this class"""

    @abstractmethod
    def train_on_batch(self, *args, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    @abstractmethod
    def evaluate_on_batch(self, *args, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    def after_backward(self, batch_idx):
        """This method is called just after `loss.backward()`"""

    def training_batch_end(self, batch_idx):
        """This method is called at the end of batch-{batch_idx}"""

    def training_end(self):
        """This method is called at the end of complete training"""

    def training_epoch_end(self, epoch, tr_metric, val_metric):
        """This method is called at the end of epoch"""
        if self.save_strategy == "epoch":
            self.save_checkpoint(os.path.join(self.output_dir, f"checkpoint-e{epoch}"))

    def __init__(self, args: TrainingArgs):
        super().__init__()

        self.output_dir = args.output_dir
        self.max_epochs = args.max_epochs

        self.precision = args.precision
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.train_batch_size = args.train_batch_size
        self.num_workers = args.num_workers

        self.save_strategy = args.save_strategy

        self.early_stop_n = args.early_stop_n
        self.epoch_saving_n = args.epoch_saving_n

        self.start_epoch = 0
        self.start_batch_idx = 0

        self.args = args

    def setup(self, model: nn.Module, tr_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, collate_fn=None):
        self.device = self.args.device

        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.precision == 'mixed16' else None

        self.args.__dict__.pop("deepspeed_plugin", None)
        wandb_args = {
            "wandb_config": self.args.__dict__,
            "project_name": self.args.project_name,
            "wandb_run_name": self.args.wandb_run_name,
            "wandb_dir": self.output_dir
        }
        self.logger = self.setup_wandb(wandb_args)

        self.tr_dataloader = self.get_dataloader(tr_dataset, collate_fn=collate_fn, is_train=True)
        self.val_dataloader = self.get_dataloader(val_dataset, collate_fn=collate_fn, is_train=False)

    def training_step(self, batch, batch_idx):
        if self.scaler is not None:
            return torch.cuda.amp.autocast()(self.train_on_batch(batch, batch_idx))
        return self.train_on_batch(batch, batch_idx)

    def validation_step(self, batch):
        return self.evaluate_on_batch(batch)

    def get_dataloader(self, dataset, collate_fn=None, is_train=True):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.train_batch_size, collate_fn=collate_fn, shuffle=is_train, pin_memory=True, num_workers=self.num_workers
        )

    def fit(
        self,
        model,
        tr_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        collate_fn=None,
        resume_from_checkpoint: str = None,
        map_location: str = "cuda:0",
    ):

        self.setup(model, tr_dataset, val_dataset, collate_fn=collate_fn)

        if resume_from_checkpoint is not None:
            print(f"Resuming from checkpoint- {resume_from_checkpoint}")
            self.load_checkpoint(resume_from_checkpoint)

        try:
            tr_metric, val_metric = self.train()
            self.display_metrics(self.max_epochs, tr_metric, val_metric)
        except KeyboardInterrupt:
            logger.warning('Interrupting through keyboard ======= Saving model weights')
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "KeyboardInterrupted-wts.bin"))

    @torch.no_grad()
    def empty_grad_(self):
        # emptying gradients in more efficient way
        for param in self.model.parameters():
            param.grad = None

    def train(self):

        tr_metric = []
        val_metric = []

        steps = 0 # updating under accumulation condition

        # setting up epochs (handling resuming)
        epochs = range(self.start_epoch, self.max_epochs)
        for epoch in epochs:

            # setting up tr_loss for accumulation
            tr_loss = 0
            losses = []

            # helping in resuming
            self.start_epoch = epoch

            # setting up progress bar to display
            desc = f"running epoch-{epoch}"
            pbar = tqdm(enumerate(self.tr_dataloader), total=len(self.tr_dataloader), desc=desc, initial=0, leave=True)
            for batch_idx, batch in pbar:
                # will help in resuming training from last-saved batch_idx
                if batch_idx != self.start_batch_idx:
                    steps += 1
                    pbar.write(f'training will start from batch_idx-{self.start_batch_idx}')
                    continue

                self.start_batch_idx += 1

                self.model.train(True)
                # simply doing forward-propogation
                loss = self.training_step(batch, batch_idx)

                if self.gradient_accumulation_steps is not None:
                    loss /= self.gradient_accumulation_steps

                # accumulating tr_loss for logging (helpful when accumulation-steps > 1)
                tr_loss += loss.item() # this should be loss.detach() if using TPUs

                # configuring for mixed-precision
                if self.scaler is not None:
                    loss = self.scaler.scale(loss)

                self.backward(loss)
                self.after_backward(batch_idx)

                self.optimizer_step(batch_idx, epoch) # handling grad_accumulation inside

                # logging with gradient accumulation handling
                if self.is_gradient_accumulation_boundary(batch_idx):

                    self.logger.log({
                    'global_steps': steps,
                    'step_tr_loss': tr_loss,
                    'learning_rate': self.optimizer.param_groups[0]["lr"],
                    }, commit=True)
                    steps += 1
                    pbar.set_postfix(tr_loss=tr_loss)

                    # accumulating losses for training-loss at epoch end
                    losses.append(tr_loss)

                    # emptying tr_loss
                    tr_loss = 0

                self.training_batch_end(batch_idx)

            # clearing batch_idx for next epoch
            self.start_batch_idx = 0

            # val_loss at training epoch end for logging
            val_loss = self.evaluate(show_progress=True)

            self.logger.log({
                'epoch': epoch,
                'tr_loss': np.mean(losses),
                'val_loss': val_loss
                }, commit=False)

            tr_metric.append(np.mean(losses))
            val_metric.append(val_loss)

            self.training_epoch_end(epoch, tr_metric, val_metric)
            if self.early_stop_n:
                self.stop_early(val_metric, self.early_stop_n, model="min")

        self.start_epoch += 1
        self.training_end()
    
        return tr_metric, val_metric

    def is_gradient_accumulation_boundary(self, batch_idx):
        return (batch_idx+1)%self.gradient_accumulation_steps == 0

    def optimizer_step(self, batch_idx, epoch):
        if self.is_gradient_accumulation_boundary(batch_idx):
            # configuring for mixed-precision
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                self.empty_grad_()
            if self.scheduler is not None:
                self.scheduler.step() # TODO: check if its correct

    def backward(self, loss):
        loss.backward()

    def evaluate(self, show_progress=True):
        # disabling layers like dropout, batch-normalization
        self.model.eval()
        running_loss = []

        desc = 'Validating ....'
        pbar = tqdm(self.val_dataloader, total=len(self.val_dataloader), desc=desc, initial=0, leave=False) if show_progress else self.val_dataloader
        for batch in pbar:
            val_loss = self.validation_step(batch)
            pbar.set_postfix(val_loss=val_loss.item())
            running_loss.append(val_loss.item())

        return np.mean(running_loss)

    def histogram_params(self, logdir="tb_params"):
        """
        You need to call this method yourself
        """
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, logdir))

        params = self.model.named_parameters()
        for n, param in params:
            writer.add_histogram(n, param)
        
        writer.close()
        # tensorboard --logdir "{tb_params}"
        
    def histogram_grads(self, logdir="tb_grads"):
        """
        You need to call this method yourself

        Remember to call this only after `.backward`
        """

        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, logdir))

        params = self.model.named_parameters()
        for n, param in params:
            if param.grad is not None:
                writer.add_histogram(n, param.grad)
            else:
                writer.add_scalar(n, 0.0)

        writer.close()
        # tensorboard --logdir "{tb_grads}"

    def save_optimizer_state_dict(self, ckpt_dir: str):
        path = os.path.join(ckpt_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), path)

    def save_training_state(self, ckpt_dir: str):
        path = os.path.join(ckpt_dir, "training_state.txt")
        with open(path, "a") as f:
            content = {
                "start_epoch": self.start_epoch,
                "start_batch_idx": self.start_batch_idx,
            }
            f.write(str(content)+"\n")

    def save_scheduler_state_dict(self, ckpt_dir: str):
        path = os.path.join(ckpt_dir, "scheduler.pt")
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), path)

    def save_model_state_dict(self, save_dir: str):
        path = os.path.join(save_dir, "pytorch_model.bin")
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), path)

    def save_checkpoint(self, ckpt_dir: str):
        ckpt_dir = self.args._setup_dir(ckpt_dir)
        self.save_model_state_dict(ckpt_dir)
        self.save_training_state(ckpt_dir)
        self.save_optimizer_state_dict(ckpt_dir)
        self.save_scheduler_state_dict(ckpt_dir)

    def load_model_state_dict(self, load_dir: str, map_location: str = "cpu"):
        """ `map_function` will be very memory expensive if you are changing the device """
        path = os.path.join(load_dir, "pytorch_model.bin")
        model = torch.load(path, map_location=map_location)
        self.model.load_state_dict(model)

    def load_checkpoint(self, ckpt_dir: str):
        raise NotImplementedError


class CONVERT_HF_DATA_TO_TORCH_DATA(torch.utils.data.Dataset):
    def __init__(self, hf_data):
        self.hf_data = hf_data
    def __len__(self):
        return len(self.hf_data)
    def __getitem__(self, idx):
        return self.hf_data[idx]

class DeepSpeedTrainer(TorchTrainer):
    def setup(self, model: nn.Module, tr_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, collate_fn=None):
        self.model = model
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = None

        wandb_args = {
            "wandb_config": self.args.__dict__,
            "project_name": self.args.project_name,
            "wandb_run_name": self.args.wandb_run_name,
            "wandb_dir": self.output_dir
        }
        self.logger = self.setup_wandb(wandb_args)

        self._init_deepspeed(tr_dataset, collate_fn=collate_fn)
        self.val_dataloader = self.get_dataloader(val_dataset)
        self.device = self.args.device = self.model.device

    def _init_deepspeed(
        self,
        tr_dataset: torch.utils.data.Dataset,
        collate_fn=None,
    ):
        ds_config = {}
        if isinstance(self.optimizer, dict):
            ds_config.update({"optimizer": self.optimizer})
            optimizer = None
        else:
            optimizer = self.optimizer

        if isinstance(self.scheduler, dict):
            ds_config.update({"scheduler": self.scheduler})
            scheduler = None
        else:
            scheduler = self.scheduler

        ds_config.update(self.args.deepspeed_plugin.__dict__)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model, optimizer, tr_dataloader, scheduler = deepspeed.initialize(
            model=self.model, model_parameters=model_parameters, config_params=ds_config,
            optimizer=optimizer, lr_scheduler=scheduler, training_data=tr_dataset, collate_fn=collate_fn,
        )

        self.optimizer = optimizer
        self.tr_dataloader = tr_dataloader
        self.scheduler = scheduler

    def is_gradient_accumulation_boundary(self, batch_idx):
        # print("GRAD ACC CHECK", self.model.is_gradient_accumulation_boundary())
        return self.model.is_gradient_accumulation_boundary()

    def optimizer_step(self, batch_idx, epoch):
        self.model.step()

    def backward(self, loss):
        self.model.backward(loss)

    def save_checkpoint(self, save_dir: str):
        client_state = {
                "start_epoch": self.start_epoch,
                "start_batch_idx": self.start_batch_idx,
            }
        self.model.save_checkpoint(save_dir, client_state=client_state)

    def load_checkpoint(self, ckpt_dir: str):
        path, client_state = self.model.load_checkpoint(ckpt_dir)
        self.start_epoch = client_state.pop("start_epoch")
        self.start_batch_idx = client_state.pop("start_batch_idx")
        return path

    def setup_optimizer(self):
        return {
            "type": "Adam",
            "params": {
            "lr": self.args.lr,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-8,
            "weight_decay": 3e-7
            }
        }

    def setup_scheduler(self):
        return {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 1.e-5,
                "warmup_max_lr": self.args.lr,
                "warmup_num_steps": 1000
            }
        }
