from typing import Callable
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from .hookbase import HookBase



# class EvalHook(HookBase):
#     """Run an evaluation function periodically.

#     It is executed every ``period`` epochs and after the last epoch.
#     """

#     def __init__(self, period: int, eval_func: Callable):
#         """
#         Args:
#             period (int): The period to run ``eval_func``. Set to 0 to
#                 not evaluate periodically (but still after the last iteration).
#             eval_func (callable): A function which takes no arguments, and
#                 returns a dict of evaluation metrics.
#         """
#         self._period = period
#         self._eval_func = eval_func

#     def _do_eval(self):
#         res = self._eval_func()
        
#         if res:
#             assert isinstance(res, dict), f"Eval function must return a dict. Got {res} instead."
#             for k, v in res.items():
#                 try:
#                     v = float(v)
#                 except Exception as e:
#                     raise ValueError(
#                         "[EvalHook] eval_function should return a dict of float. "
#                         f"Got '{k}: {v}' instead."
#                     ) from e
#             self.log(self.trainer.epoch, **res, smooth=False)

#     def after_epoch(self):
#         if self.every_n_epochs(self._period) or self.is_last_epoch():
#             self._do_eval()




class EvalHook(HookBase):
    """Run an evaluation function periodically."""
    def __init__(self, period:int=None,  **kwargs):
        """
        Args:
            period (int): The period to run evaluate, iter based.

        """
        self._period = period



    def after_epoch(self):
        eval_metrics = self._do_eval()
        self.log(self.trainer.cur_iter, **eval_metrics) # cur_iter是全局递增的，包含了所有epoch



    def _do_eval(self):
        "一般针对不同任务，需要根据情况改写"
        correct = 0
        eval_loss = 0
        eval_metrics = {}  # 保存需要记录的指标

        self.trainer.model.eval()

        # 这部分会根据网络io不同而具体改写
        with torch.no_grad():
            for i, batch in enumerate(self.trainer.eval_data_loader):
                inputs, target = batch
                output = self.trainer.model(inputs)
                eval_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()

        eval_loss = eval_loss/ len(self.trainer.eval_data_loader.dataset)
        eval_acc = 100. * correct / len(self.trainer.eval_data_loader.dataset)
    
        eval_metrics['eval_loss'] = eval_loss
        eval_metrics['eval_acc'] = eval_acc
        eval_metrics['smooth'] = False # 是否需要进行平滑，tensorboard可以在前端做平滑

        self.trainer.model.train()
        return eval_metrics

    
    def after_iter(self) -> None:
        if self._period and self.every_n_inner_iters(self._period):
            eval_metrics = self._do_eval()
            self.log(self.trainer.cur_iter, **eval_metrics)