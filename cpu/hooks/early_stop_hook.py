from logging import lastResort, raiseExceptions
import torch
from .hookbase import HookBase

class EarlyStopHook(HookBase):
    "Monitor a metric and stop training when it stops improving."

    def __init__(self, monitor:str, patience:int):
        super().__init__()
        self.patience = patience  # epoch based
        self.monitor = monitor
        self.best_monitor = 0
        self.last_improve_epoch = 0

    
    def after_iter(self): # 每次iter后记录该epoch有没有提升
        if self.is_eval_iter():
            cur_monitor = self.metric_storage[self.monitor].latest 
            if cur_monitor > self.best_monitor:
                self.last_improve_epoch = self.cur_epoch
                self.best_monitor = cur_monitor

    def after_epoch(self):
        cur_monitor = self.metric_storage[self.monitor].latest 
        if cur_monitor > self.best_monitor:
            self.last_improve_epoch = self.cur_epoch
            self.best_monitor = cur_monitor

        if self.cur_epoch - self.last_improve_epoch > self.patience:
            self.trainer._early_stop_flag = True
        
            
            


        