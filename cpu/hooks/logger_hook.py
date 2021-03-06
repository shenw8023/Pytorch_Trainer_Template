import datetime
import logging
import time
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from .hookbase import HookBase

logger = logging.getLogger(__name__)



class LoggerHook(HookBase):
    """Write metrics to console and tensorboard files."""

    def __init__(self, period: int = 50, tb_log_dir: str = "log_dir", **kwargs) -> None:
        """
        Args:
            period (int): The period to write metrics. Defaults to 50.
            tb_log_dir (str): The directory to save the tensorboard files. Defaults to "log_dir".
            kwargs: Other arguments passed to ``torch.utils.tensorboard.SummaryWriter(...)``
        """
        self._period = period
        self._tb_writer = SummaryWriter(tb_log_dir, **kwargs)
        # metric name -> the latest iteration written to tensorboard file
        self._last_write: Dict[str, int] = {}  # 记录最新一次写入的iter号，用于判断避免重复写入

    def before_train(self) -> None:
        self._train_start_time = time.perf_counter()

    def after_train(self) -> None:
        self._tb_writer.close()
        total_train_time = time.perf_counter() - self._train_start_time
        try:
            total_hook_time = total_train_time - self.metric_storage["iter_time"].global_sum   # 完整训练时间 - 所有iter计算耗时
            logger.info(
                "Total training time: {} ({} on hooks)".format(
                    str(datetime.timedelta(seconds=int(total_train_time))),
                    str(datetime.timedelta(seconds=int(total_hook_time))),
                )
            )
        except KeyError:
            pass

    def after_epoch(self) -> None:
        # Some hooks maybe generate logs in after_epoch().
        # When LoggerHook is the last hook, calling _write_tensorboard()
        # at every after_epoch() can avoid missing logs.
        self._write_console_eval()
        self._write_tensorboard()  # 部分指标在epoch后产生，所以每个epoch后还要写一次，为防止在after_iter中已经写过，重复写入，维护一个最新iter号用于判断




    def _write_console_eval(self):
        space = "\t"
        process_string = f"Epoch: [{self.trainer.epoch}][{self.trainer.inner_iter+1}/{self.trainer.epoch_len}]"
        lr = self.metric_storage["lr"].latest if "lr" in self.metric_storage else None
        lr_strings = f"lr: {lr:.3f}" if lr is not None else ""

        train_loss = self.metric_storage["train_loss"].latest
        train_loss_strings = f"train_loss: {train_loss:.5f}"

        eval_loss =  self.metric_storage["eval_loss"].latest
        eval_loss_string = f"eval_loss: {eval_loss:.5f}"

        acc = self.metric_storage['eval_acc'].latest
        acc_string = f"eval_acc: {acc:.3f}%"

        logger.info(
            "{process}{train_loss}{eval_loss}{acc}{lr}".format(
                process=process_string,
                train_loss=space + train_loss_strings,
                eval_loss=space + eval_loss_string,
                acc=space + acc_string,
                lr=space + lr_strings,
                )
            )
        
    def _write_console_train(self):
        space = "\t"
        process_string = f"Epoch: [{self.trainer.epoch}][{self.trainer.inner_iter+1}/{self.trainer.epoch_len}]"
        lr = self.metric_storage["lr"].latest if "lr" in self.metric_storage else None
        lr_strings = f"lr: {lr:.3f}" if lr is not None else ""
        train_loss = self.metric_storage["train_loss"].latest
        train_loss_strings = f"train_loss: {train_loss:.5f}"
        
        logger.info(
            "{process}{train_loss}{lr}".format(
                process=process_string,
                train_loss=space + train_loss_strings,
                lr=space + lr_strings,
            )
        )



    def _write_console(self) -> None:
        # These fields ("data_time", "iter_time", "lr", "loss") may does not
        # exist when user overwrites `self.trainer.train_one_iter()`
        data_time = self.metric_storage["data_time"].avg if "data_time" in self.metric_storage else None
        iter_time = self.metric_storage["iter_time"].avg if "iter_time" in self.metric_storage else None
        lr = self.metric_storage["lr"].latest if "lr" in self.metric_storage else None

        if iter_time is not None:
            eta_seconds = iter_time * (self.trainer.max_iters - self.trainer.cur_iter - 1)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_string = None

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        loss_strings = [
            f"{key}: {his_buf.avg:.4g}"
            for key, his_buf in self.metric_storage.items()
            if "loss" in key
        ]

        process_string = f"Epoch: [{self.trainer.epoch}][{self.trainer.inner_iter+1}/{self.trainer.epoch_len}]"  # +1是因为inner_iter是从0开始的

        # space = " " * 2
        space = "\t"
        logger.info(
            "{process}{eta}{losses}{iter_time}{data_time}{lr}{memory}".format(
                process=process_string,
                eta=space + f"ETA: {eta_string}" if eta_string is not None else "",
                losses=space + "\t".join(loss_strings) if loss_strings else "",
                iter_time=space + f"iter_time: {iter_time:.4f}" if iter_time is not None else "",
                data_time=space + f"data_time: {data_time:.4f}  " if data_time is not None else "",
                lr=space + f"lr: {lr:.5g}" if lr is not None else "",
                memory=space + f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
            )
        )

    def _write_tensorboard(self) -> None:
        for key, (iter, value) in self.metric_storage.values_maybe_smooth.items():
            if key not in self._last_write or iter > self._last_write[key]:  # 防止重复写入
                self._tb_writer.add_scalar(key, value, iter)
                self._last_write[key] = iter

    # def after_iter(self) -> None:
    #     if self.every_n_inner_iters(self._period):
    #         self._write_console_train()
    #         self._write_tensorboard()

    def after_iter(self) -> None:
        if self.every_n_inner_iters(self._period):
            self._write_console_train()
            self._write_tensorboard()
            
        if self.trainer._eval_period and self.every_n_inner_iters(self.trainer._eval_period):
            self._write_console_eval()
        