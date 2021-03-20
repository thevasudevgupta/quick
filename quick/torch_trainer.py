# __author__ = 'Vasudev Gupta'
# __author_email__ = '7vasudevgupta@gmail.com'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import wandb
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    logger.warning("CUDA is not available => Training will happen on CPU")

"""
USAGE:

    >>> import quick

    >>> class Trainer(quick.Trainer):
            def __init__(self, args: TrainingArgs):
                super().__init__(args)

            def setup_optimizer(self):
                '''
                    ....
                '''
                return

            def train_batch(self, batch, batch_idx):
                '''
                    ....
                '''
                return

            def validate_batch(self, batch):
                '''
                    ....
                '''
                return

    >>> args = quick.TorchTrainingArgs()
    >>> trainer = quick.TorchTrainer(args)
    >>> trainer.setup(model)

    >>> trainer.fit(tr_dataset, val_dataset)
"""

@dataclass
class TorchTrainingArgs:

    lr: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_epochs: int = 5

    base_dir: str = None
    save_epoch_dir: str = None

    project_name: str = "Quick-project"
    wandb_run_name: str = None

    map_location: str = "cuda:0"
    early_stop_n: int = None
    epoch_saving_n: int = 3

    precision: str = "float32"

    def __post_init__(self):

        self.base_dir = "." if self.base_dir is None else self.base_dir
        self.map_location = torch.device(self.map_location)

        if self.precision == "mixed16":
            logger.wanrning("mixed precision training is not supported curreltly, Setting `precision='mixed16'`")

        print("model weights will be saved in {path} file".format(path=os.path.join(self.base_dir, "pytorch_model.bin")))
        # training stuff will be in `training.tar`


class TrainerSetup(object):

    def __init__(self):
        """
        This class is mainly having setup methods for enable training
        """

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
    def _sanity_check(args: TorchTrainingArgs):
        if not isinstance(args, TorchTrainingArgs):
            raise ValueError("Your argument class must be inherited from TrainingArgs")

    @staticmethod
    def _setup_savedir(save_dir: str, base_dir: str):
        if save_dir:
            if save_dir not in os.listdir(base_dir):
                os.mkdir(os.path.join(base_dir, save_dir))
            return save_dir
        return save_dir

    @staticmethod
    def _setup_basedir(base_dir: str):
        if base_dir is None:
            return "."
        elif base_dir == ".":
            return base_dir
        elif base_dir not in os.listdir():
            os.mkdir(base_dir)
            return base_dir

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

    @abstractmethod
    def setup_optimizer(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    def setup_scheduler(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    @abstractmethod
    def train_on_batch(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    @abstractmethod
    def evaluate_on_batch(self, **kwargs):
        """This method must be implemented in the class inherited from this class"""

    def after_backward(self, batch_idx):
        """This method is called just after `loss.backward()`"""

    def training_batch_end(self, batch_idx):
        """This method is called at the end of batch-{batch_idx}"""

    def training_end(self):
        """This method is called at the end of complete training"""

    def training_epoch_end(self, epoch, tr_metric, val_metric):
        """This method is called at the end of epoch"""
        save_status = None
        if self.save_epoch_dir:
            save_status = self.assert_epoch_saving(val_metric, n=self.epoch_saving_n, mode="min")
        if save_status:
            self.save_model_state_dict(os.path.join(self.base_dir, self.save_epoch_dir+f"-e{epoch}"))
            self.save_training_state_dict(self.base_dir)

    def __init__(self, args: TorchTrainingArgs):
        super().__init__()

        self._sanity_check(args)

        self.map_location = args.map_location
        self.max_epochs = args.max_epochs

        self.precision = args.precision
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.base_dir = args.base_dir
        self.save_epoch_dir = args.save_epoch_dir

        self.batch_size = args.batch_size

        self.early_stop_n = args.early_stop_n
        self.epoch_saving_n = args.epoch_saving_n

        self.start_epoch = 0
        self.start_batch_idx = 0

        self.args = args

    def setup(self, model: nn.Module):

        self.base_dir = self._setup_basedir(self.base_dir)
        self.save_epoch_dir = self._setup_savedir(self.save_epoch_dir, self.base_dir)

        self.model = model
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.scaler = self.setup_scaler()

        self.device = self.setup_hardware()

        print(f"[DEVICE:] {self.device}")

        wandb_args = {
            "wandb_config": self.args,
            "project_name": self.args.project_name,
            "wandb_run_name": self.args.wandb_run_name,
            "wandb_dir": self.base_dir
        }

        self.setup_wandb(wandb_args)

    def setup_scaler(self):
        if self.precision == 'mixed16':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available')
            logger.info('Training with mixed16')
            return torch.cuda.amp.GradScaler()

    def setup_hardware(self):
        device = torch.device('cpu')

        if torch.cuda.is_available():
            device = torch.device("cuda")

        return device

    def training_step(self, batch, batch_idx):

        if self.precision == 'mixed16':
            return torch.cuda.amp.autocast(self.train_on_batch)(batch, batch_idx)

        return self.train_on_batch(batch, batch_idx)

    def validation_step(self, batch):
        return self.evaluate_on_batch(batch)

    def fit(
        self,
        tr_dataset: torch.utils.data.DataLoader,
        val_dataset: torch.utils.data.DataLoader
    ):

        try:
            tr_metric, val_metric = self.train(tr_dataset, val_dataset)            
            self.display_metrics(self.max_epochs, tr_metric, val_metric)

        except KeyboardInterrupt:
            logger.warning('Interrupting through keyboard ======= Saving model weights')
            self.save_model_state_dict(os.path.join(self.base_dir, "keyboard-interrupted-weights"))

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
            pbar = tqdm(enumerate(tr_dataset), total=len(tr_dataset), desc=desc, initial=0, leave=False)
            for batch_idx, batch in pbar:
                # will help in resuming training from last-saved batch_idx
                if batch_idx != self.start_batch_idx:
                    steps += 1
                    pbar.write(f'training will start from batch_idx-{self.start_batch_idx}')
                    continue

                self.start_batch_idx += 1

                self.model.train(True)
                # simply doing forward-propogation
                loss = self.train_step(batch, batch_idx)
                loss /= self.gradient_accumulation_steps

                # accumulating tr_loss for logging (helpful when accumulation-steps > 1)
                tr_loss += loss.item()

                # configuring for mixed-precision
                if self.precision == 'mixed16':
                    loss = self.scaler.scale(loss)

                loss.backward()

                self.after_backward(batch_idx)

                # gradient accumulation handler
                if (batch_idx+1)%self.gradient_accumulation_steps == 0:

                    self.optimizer_step(batch_idx, epoch)

                    wandb.log({
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

            wandb.log({
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

    def optimizer_step(self, batch_idx, epoch):
        # configuring for mixed-precision
        if self.precision == 'mixed16':
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            self.empty_grad_()
            # self.scheduler.step(batch_idx) # TODO: check if its correct

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
        writer = SummaryWriter(log_dir=os.path.join(self.base_dir, logdir))

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

        writer = SummaryWriter(log_dir=os.path.join(self.base_dir, logdir))

        params = self.model.named_parameters()
        for n, param in params:
            if param.grad is not None:
                writer.add_histogram(n, param.grad)
            else:
                writer.add_scalar(n, 0.0)

        writer.close()
        # tensorboard --logdir "{tb_grads}"

    def save_training_state_dict(self, save_dir: str):

        path = os.path.join(save_dir, "training.tar")

        # defining what all to save
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'start_epoch': self.start_epoch,
            'start_batch_idx':  self.start_batch_idx,
            }

        if self.scheduler is not None:
            state_dict.update({
                "scheduler": self.scheduler.state_dict(),
            })

        torch.save(state_dict, path)

    def save_model_state_dict(self, save_dir: str):

        path = os.path.join(save_dir, "pytorch_model.bin")
        state_dict = self.model.state_dict()

        torch.save(state_dict, path)

    def load_model_state_dict(self, load_dir: str):

        path = os.path.join(load_dir, "pytorch_model.bin")
        """
        Note:
            `map_function` will be very memory expensive if you are changing the device
        """

        print(
            """loading:
                1) model state_dict
            """
        )

        model = torch.load(path, map_location=self.map_location)
        self.model.load_state_dict(model)        

    def load_training_state_dict(self, load_dir: str):
        
        path = os.path.join(load_dir, "training.tar")

        print(
            """Loading:
                1) optimizer state_dict
                2) scheduler state_dict
                3) start_epoch
                4) start_batch_idx
            """
            )

        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

        # helpful in resuming training from particular step
        self.start_epoch = checkpoint.pop('start_epoch')
        self.start_batch_idx = checkpoint.pop('start_batch_idx')

        print(f'loading successful (start-epoch-{self.start_epoch}, start_batch_idx-{self.start_batch_idx})')
