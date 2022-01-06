import os
import os.path as osp
from typing import Any, Dict, List, Optional

from .hookbase import HookBase


class CheckpointerHook(HookBase):
    """Save checkpoints periodically.

    Save checkpoint, if current epoch is a multiple of period or ``max_epochs`` is reached.
    """

    def __init__(self, period: int, max_to_keep: Optional[int] = None) -> None:
        """
        Args:
            period (int): The period to save checkpoint.
            max_to_keep (int): Maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted.
        """
        self._period = period
        assert max_to_keep is None or max_to_keep > 0
        self._max_to_keep = max_to_keep
        self._recent_checkpoints: List[str] = []


    def after_iter(self):
        if self.is_eval_iter():
            self.save_checkpoint()
            

    def after_epoch(self):
        self.save_checkpoint()

    def save_checkpoint(self):
        monitor_name = self.trainer._early_stop_monitor
        cur_monitor = self.metric_storage[monitor_name].latest
        if cur_monitor > self.trainer._best_monitor:  # 根据当前最新指标和历史最优指标判断是否保存
            self.trainer._best_monitor = cur_monitor  # 更新历史最优指标
            epoch = self.trainer.epoch
            checkpoint_name = f"epoch-{epoch}-{monitor_name}-{self.trainer._best_monitor}.pth"
            self.trainer.save_checkpoint(checkpoint_name)

            if self._max_to_keep is not None:
                self._recent_checkpoints.append(checkpoint_name)
                if len(self._recent_checkpoints) > self._max_to_keep:
                    # delete the oldest checkpoint
                    file_name = self._recent_checkpoints.pop(0)
                    file_path = osp.join(self.trainer.ckpt_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

        


    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "trainer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
