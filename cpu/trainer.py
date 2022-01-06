import logging
import os
import os.path as osp
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cpu.hooks import early_stop_hook

from .hooks import CheckpointerHook, HookBase, LoggerHook, EvalHook, EarlyStopHook
from .logger import setup_logger
from .lr_scheduler import LRWarmupScheduler
from .history_buffer import HistoryBuffer
from .misc import symlink

logger = logging.getLogger(__name__)


class Trainer:
    """An epoch-based trainer.

    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source epoch-based optimization
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    4. Adjust the learning rate.

    All other tasks during training (checkpointing, logging, evaluation) are maintained
    by hooks, which can be registered by :meth:`register_hooks`.

    If you want to do anything fancier than this, either subclass this class
    and implement your own :meth:`train_one_iter`, or write your own trainer.

    .. code-block:: python

        # create your model / optimizer / lr_scheduler / data_loader before using the trainer
        model = ...
        optimizer = ...
        lr_scheduler = ...
        data_loader = ...
        # train 100 epochs
        trainer = Trainer(model, optimizer, lr_scheduler, data_loader, max_epochs=100)
        trainer.train()

    .. Note::

        Currently only support single GPU training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        train_data_loader: DataLoader,
        eval_data_loader: DataLoader,
        max_epochs: int,
        work_dir: str = "work_dir",
        max_num_checkpoints: int = None,
        eval_period: int = None,      # (iter-based)  如果没有指定的话，就在每个epoch后do_eval
        checkpoint_period: int = 1,   # (epoch-based)
        log_period: int = 100,        # (iter-based)
        clip_grad_norm: float = 0.0,
        enable_amp=False,
        warmup_method: Optional[str] = None,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
        early_stop_patience: int = 3,
        early_stop_monitor: str = "eval_acc"
    ):
        """
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            lr_scheduler (optim.lr_scheduler._LRScheduler)
            train_data_loader (torch.utils.data.DataLoader): Training data loader.
            eval_data_loader (torch.utils.data.DataLoader): eval data loader.
            max_epochs (int): Total training epochs.
            work_dir (str): The working directory to save checkpoints and logs.
                Defaults to "work_dir".
            max_num_checkpoints (int): The maximum number of checkpoints to save.
                If None, save all checkpoints. Defaults to None.
            eval_period (int): The period (iter-based) to do evaluate. Defaults to 50.
            checkpoint_period (int): The period (epoch-based) to save checkpoint. Defaults to 1.
            log_period (int): The period (iter-based) to log. Defaults to 50.
            clip_grad_norm (float): Max norm of the gradients. If <= 0, will not clip gradients.
                Defaults to 0.
            enable_amp (bool): Enable the Automatic Mixed Precision (AMP) training.
                Defaults to False.
            warmup_method (str): Type of warmup used. It can be None (no warmup),
                "constant", "linear" or "exp". Defaults to None.
            warmup_iters (int): The number of iterations that warmup lasts. Defaults to 1000.
            warmup_factor (float): LR used at the beginning of warmup equals to
                ``warmup_factor * initial_lr``. Defaults to 0.001.
        """
        self.model = model
        self.optimizer = optimizer
        # convert epoch-based scheduler to iteration-based scheduler
        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler, len(train_data_loader), warmup_method, warmup_iters, warmup_factor
        )
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.work_dir = work_dir
        self.metric_storage = MetricStorage()

        # counters
        self.inner_iter: int  # [0, epoch_len - 1]
        self.epoch: int  # [0, max_epochs-1]
        self.start_epoch = 0  # [0, max_epochs-1]
        self.max_epochs = max_epochs

        self._hooks: List[HookBase] = []
        self._data_iter_train = iter(train_data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._eval_period = eval_period
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp
        self._early_stop_flag = False
        self._early_stop_patience = early_stop_patience
        self._early_stop_monitor = early_stop_monitor

        self.register_hooks(self._build_default_hooks())
        logger.info(f"Registered default hooks: {self.registered_hook_names}")  # TODO 是因为模块被调用没有执行吗

        if self._enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")
            self._grad_scaler = GradScaler()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def epoch_len(self) -> int:
        return len(self.train_data_loader)

    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    @property
    def cur_iter(self) -> int:
        """Returns the current iteration ranged in [0, max_iters - 1]."""
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        """The iteration to start from. The minimum possible value is 0."""
        return self.start_epoch * self.epoch_len

    @property
    def ckpt_dir(self) -> str:
        return osp.join(self.work_dir, "checkpoints")

    @property
    def tb_log_dir(self) -> str:
        return osp.join(self.work_dir, "tb_logs")

    @property
    def log_file(self) -> str:
        return osp.join(self.work_dir, "log.txt")

    @property
    def model_or_module(self) -> nn.Module:
        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            return self.model.module
        return self.model

    @property
    def registered_hook_names(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    def log(self, *args, **kwargs) -> None:
        """Update metrics."""
        self.metric_storage.update(*args, **kwargs)

    def _prepare_for_training(self) -> None:
        # setup the root logger of the `cpu` library to show
        # the log messages generated from this library
        setup_logger("cpu", output=self.log_file)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        split_line = "-" * 50
        logger.info(
            f"\n{split_line}\n"
            f"Work directory: {self.work_dir}\n"
            f"Checkpoint directory: {self.ckpt_dir}\n"
            f"Tensorboard directory: {self.tb_log_dir}\n"
            f"Log file: {self.log_file}\n"
            f"{split_line}"
        )

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer.

        The hooks are executed in the order they are registered.

        Args:
            hooks (list[HookBase]): List of hooks.
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other. This normally
            # does not matter, but will cause memory leak if the involved objects contain __del__.
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
            # We always keep :class:`LoggerHook` as the last hook to avoid losing any records
            # that should have been logged. The order of other hooks remains the same.
            if self._hooks and isinstance(self._hooks[-1], LoggerHook):
                self._hooks.insert(len(self._hooks) - 1, h)
            else:
                self._hooks.append(h)

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _build_default_hooks(self) -> List[HookBase]:  # 顺序是有讲究的
        default_hooks =  [
            EvalHook(self._eval_period),
            EarlyStopHook(self._early_stop_monitor, self._early_stop_patience),
            CheckpointerHook(self._checkpoint_period, self._max_num_checkpoints),
            LoggerHook(self._log_period, tb_log_dir=self.tb_log_dir),
            
        ]
        return default_hooks

    def _log_iter_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float,
                          iter_time: float, lr: float) -> None:
        """
        Args:
            loss_dict (dict): Dict of scalar losses.
            data_time (float): Time taken by the dataloader iteration.
            iter_time (float): Time taken by one complete iteration.
            lr (float): Learning rate used in this iteration.
        """
        # 记录时间指标
        self.log(self.cur_iter, data_time=data_time, iter_time=iter_time)   # TODO 只有进入tensorboard的才涉及smooth与否
        # 记录lr指标
        self.log(self.cur_iter, lr=lr, smooth=False)

        loss_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        loss_value = sum(loss_dict.values())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
            )

        # 记录sum loss指标
        self.log(self.cur_iter, total_loss=loss_value) 
        # 记录单个所有loss
        if len(loss_dict) > 1:
            self.log(self.cur_iter, **loss_dict)


    def train_one_iter(self) -> None:
        """Train one iteration.

        .. Note::

            Standard PyTorch LR scheduler is epoch-based and called at the end of epoch.
            However, our scheduler is iteration-based, so it should be called after every iteration.

        Subclass :class:`cpu.Trainer` and implement your :meth:`train_one_iter`
        to do something fancier.
        """
        iter_start_time = time.perf_counter()
        lr_this_iter = self.lr

        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        batch = next(self._data_iter_train)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        if self._enable_amp:
            with autocast():
                loss_dict = self.model(batch)  # 这种情况要求model forward返回的是loss，其他情况（例如返回logits）需要重写该方法
        else:
            loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())   

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        if self._enable_amp:
            self._grad_scaler.scale(losses).backward()
        else:
            losses.backward()
        if self._clip_grad_norm > 0:
            if self._enable_amp:
                self._grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        ##############################
        # 4. Update model parameters #
        ##############################
        if self._enable_amp:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()

        ###########################
        # 5. Adjust learning rate #
        ###########################
        self.lr_scheduler.step()

        self._log_iter_metrics(loss_dict, data_time, time.perf_counter() - iter_start_time, lr_this_iter)

    def _train_one_epoch(self) -> None:
        # evaluation hook changes the model to `eval` mode after finishing epoch
        self.model.train()
        for self.inner_iter in range(self.epoch_len):  # self.inner_iter这个全局变量
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
        # update data iterator to avoid `StopIteration` exception in the next epoch
        self._data_iter_train = iter(self.train_data_loader)


    def train(self) -> None:
        """Start training."""
        logger.info(f"Start training from epoch {self.start_epoch}")  # TODO 没有执行
        self._prepare_for_training()  # set logger
        self._call_hooks("before_train")
        for self.epoch in range(self.start_epoch, self.max_epochs+1):   # self.epoch这个变量全局变化
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
            if self._early_stop_flag:
                logger.info(f"no optimization for {self._early_stop_patience} epochs, auto stop")
                break
        self._call_hooks("after_train")

    def test(self) -> None:
        pass # TODO


    def save_checkpoint(self, file_name: str) -> None:
        """Save "epoch", "model", "optimizer", "lr_scheduler", "metric_storage",
        "hooks" (optional), "grad_scaler" (optional).

        Args:
            filename (str): The name of the file to save.
        """
        data = {
            "epoch": self.epoch,
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
        }
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states
        if self._enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = osp.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag last checkpoint
        dst_file = osp.join(self.ckpt_dir, "latest.pth")  # 每次一个epoch出来以后都软链接记录一下最新保存模型
        symlink(file_name, dst_file)

    def load_checkpoint(self, path: str = "", checkpoint: Dict[str, Any] = None):
        """Load the given checkpoint.

        Args:
            checkpoint (dict): The checkpoint to load.
            path (str): Path to the checkpoint. If empty, will not load anything.
                `checkpoint` and `path` can only be specified one.
        """
        assert (checkpoint is not None) ^ (path != "")
        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")

        # 1. load epoch
        self.start_epoch = checkpoint["epoch"] + 1

        # 2. load metric_storage
        self.metric_storage = checkpoint["metric_storage"]

        # 3. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 4. load lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 5. load grad scaler
        consistent_amp = not (self._enable_amp ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self._enable_amp:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 6. load model
        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning("Encounter missing keys when loading model weights:\n"
                            f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning("Encounter unexpected keys when loading model weights:\n"
                            f"{incompatible.unexpected_keys}")

        # 7. load hooks
        hook_states = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_states]
        unexpected_keys = [key for key in hook_states if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}")

        for key, value in hook_states.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break


class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    The class is designed for automatic tensorboard logging. User should specify the ``smooth``
    when calling :meth:`update`, in order to we can determine which metrics should be
    smoothed when performing tensorboard logging.

    # 记录各项指标在训练过程中值的变化

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}     # 默认smooth返回一段窗口内均值，否则返回该指标最新值
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[str, HistoryBuffer] = self
        self._smooth: Dict[str, bool] = {}  # 记录每个指标是否要进行smooth
        self._latest_iter: Dict[str, int] = {}  # 记录每个指标最新iter索引

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = HistoryBuffer(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        Returns:
            dict[str -> (int, float)]: Mapping from metric name to its
                (the latest iteration, the avg/latest value) pair.
        """
        return {
            key: (self._latest_iter[key], his_buf.avg if self._smooth[key] else his_buf.latest) # 如果该指标需要smooth，返回avg，否则返回latest
            for key, his_buf in self._history.items()
        }
