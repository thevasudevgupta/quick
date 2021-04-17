# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'

# TODO:
# gradient accumulation deepspeed
# 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import wandb
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


try:
    import deepspeed
    DEEPSPEED_INSTALLATION_STATUS = True
except:
    logger.warning("DeepSpeed is not available => run `pip3 install deepspeed` if you want to use DeepSpeed")
    DEEPSPEED_INSTALLATION_STATUS = False

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

    ...     def evaluate_on_batch(self, batch):
    ...         batch = {k: batch[k].to(self.device) for k in batch}
    ...         out = self.model(**batch, return_dict=True)
    ...         return out["loss"].mean()

    >>> from quick import TrainingArgs
    >>> args = TrainingArgs(enable_deepspeed=ENABLE_DEEPSPEED)
    >>> trainer = Trainer(args)
    >>> trainer.setup(model)

    >>> trainer.fit(tr_dataset, val_dataset)
"""

@dataclass
class DeepSpeedPlugin:
    # this will take all value you would have specified in `ds_config.json`

    local_rank: int  = 0
    train_batch_size: int = None
    gradient_accumulation_steps: int = None

    fp16: tuple = (
        ("enabled", True),
    )

    zero_optimization: tuple = (
        ("stage", 2),
        ("cpu_offload", False),
    )

    def __post_init__(self):
        for attr_name in ["fp16", "zero_optimization"]:
            if isinstance(getattr(self, attr_name), tuple):
                temp = {}
                for k, v in getattr(self, attr_name):
                    temp[k] = v
                setattr(self, attr_name, temp)
            assert isinstance(getattr(self, attr_name), dict), f"{attr_name} must be dict"


@dataclass
class TrainingArgs:

    # unused
    batch_size: int = 8
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
        if not torch.cuda.is_available():
            logger.warning("[Quick WARNING] CUDA is not available => Training will happen on CPU")

        self.output_dir = self._setup_dir(self.output_dir)

        if self.precision == "mixed16":
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            raise NotImplementedError

        if self.save_strategy is None:
            logger.warning("[Quick WARNING] You are not saving anything")

        # this will be overwritten in DeepSpeedTrainer.setup()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

        # setting up deepspeed plugin
        self.deepspeed_plugin = replace(
            self.deepspeed_plugin,
            train_batch_size=self.batch_size * self.gradient_accumulation_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        if self.enable_deepspeed:
            self.precision = None
            self.batch_size = None
            self.gradient_accumulation_steps = None
            self.device = None
            assert DEEPSPEED_INSTALLATION_STATUS, "DeepSpeed not installed => Run `pip3 install deepspeed`"

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

        self.save_strategy = args.save_strategy

        self.early_stop_n = args.early_stop_n
        self.epoch_saving_n = args.epoch_saving_n

        self.start_epoch = 0
        self.start_batch_idx = 0

        self.args = args

    def setup(self, model: nn.Module):
        self.device = self.args.device

        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        else:
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

    def training_step(self, batch, batch_idx):
        if self.scaler is not None:
            return torch.cuda.amp.autocast(self.train_on_batch)(batch, batch_idx)
        return self.train_on_batch(batch, batch_idx)

    def validation_step(self, batch):
        return self.evaluate_on_batch(batch)

    def fit(
        self,
        tr_dataset: torch.utils.data.DataLoader,
        val_dataset: torch.utils.data.DataLoader,
        checkpoint_dir: str = None,
        map_location: str = "cuda:0",
    ):

        if checkpoint_dir is not None:
            print(f"Resuming from checkpoint- {checkpoint_dir}")
            self.load_checkpoint(checkpoint_dir)

        try:
            tr_metric, val_metric = self.train(tr_dataset, val_dataset)
            self.display_metrics(self.max_epochs, tr_metric, val_metric)
        except KeyboardInterrupt:
            logger.warning('Interrupting through keyboard ======= Saving model weights')
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "KeyboardInterrupted-wts.bin"))

    @torch.no_grad()
    def empty_grad_(self):
        # emptying gradients in more efficient way
        for param in self.model.parameters():
            param.grad = None

    def train(self, tr_dataset, val_dataset):

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
            pbar = tqdm(enumerate(tr_dataset), total=len(tr_dataset), desc=desc, initial=0, leave=True)
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

                # gradient accumulation handler
                if self.is_gradient_accumulation_boundary(batch_idx):
                    self.optimizer_step(batch_idx, epoch)

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
            val_loss = self.evaluate(val_dataset, show_progress=True)

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

    def evaluate(self, val_dataset, show_progress=True):
        # disabling layers like dropout, batch-normalization
        self.model.eval()
        running_loss = []

        desc = 'Validating ....'
        pbar = tqdm(val_dataset, total=len(val_dataset), desc=desc, initial=0, leave=False) if show_progress else val_dataset
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


class DeepSpeedTrainer(TorchTrainer):

    def setup(self, model: nn.Module):

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

        self.model, self.optimizer, self.scheduler = self.init_deepspeed(model)
        self.device = self.args.device = model.device

    def init_deepspeed(
        self,
        model: nn.Module,
    ):
        ds_config = {}
        if isinstance(self.optimizer, dict):
            ds_config.update({"optimizer": self.optimizer})
        else:
            raise NotImplementedError
        if isinstance(self.scheduler, dict):
            ds_config.update({"scheduler": self.scheduler})
        else:
            raise NotImplementedError
        ds_config.update(self.args.deepspeed_plugin.__dict__)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model, model_parameters=model_parameters, config_params=ds_config,
        )

        return model, optimizer, scheduler

    def is_gradient_accumulation_boundary(self, batch_idx):
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
