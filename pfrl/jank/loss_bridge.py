from abc import ABC, abstractmethod
from typing import List, Dict

import torch


class AbstractLossBridge(ABC):
    @abstractmethod
    def get_loss(self, batch_state, batch_action, batch_updated_action, batch_reward, batch_next_state, update_interval) -> torch.Tensor:
        raise NotImplementedError()


class StaticLossBridge(AbstractLossBridge):
    def __init__(self):
        self.loss_got_got = None
        self.loss = None

    def set(self, loss: torch.Tensor):
        self.loss = loss
        self.loss_got_got = False

    def get_loss(self, batch_state, batch_action, batch_updated_action, batch_reward, batch_next_state, update_interval) -> torch.Tensor:
        if self.loss_got_got is None:
            raise Exception("You must set the loss before getting in, dummy.")
        if self.loss_got_got:
            raise Exception("Cannot access the loss in a LossBridge more than once.")
        assert self.loss is not None, "If you're trying to get a None loss, you messed up somewhere."

        self.loss_got_got = True
        return self.loss


class NoLossBridge(AbstractLossBridge):
    def get_loss(self, batch_state, batch_action, batch_updated_action, batch_reward, batch_next_state, update_interval):
        return 0
