from abc import ABC, abstractmethod
from typing import List, Dict

import torch


class AbstractLossBridge(ABC):
    @abstractmethod
    def get(self, batch_state, batch_action, batch_reward, batch_next_state) -> torch.Tensor:
        raise NotImplementedError()


class StaticLossBridge(AbstractLossBridge):
    def __init__(self):
        self.loss_got_got = None
        self.loss = None

    def set(self, loss: torch.Tensor):
        self.loss = loss
        self.loss_got_got = False

    def get(self, batch_state, batch_action, batch_reward, batch_next_state) -> torch.Tensor:
        if self.loss_got_got is None:
            raise Exception("You must set the loss before getting in, dummy.")
        if self.loss_got_got:
            raise Exception("Cannot access the loss in a LossBridge more than once.")
        assert self.loss is not None, "If you're trying to get a None loss, you messed up somewhere."

        self.loss_got_got = True
        return self.loss


class LossCalculator(ABC):
    @abstractmethod
    def __call__(self, batch_state, batch_action, batch_reward, batch_next_state):
        raise NotImplementedError()


class DynamicLossBridge(AbstractLossBridge):
    def __init__(self, loss_calculator: LossCalculator):
        super().__init__()
        self.loss_calculator = loss_calculator

    def get(self, batch_state, batch_action, batch_reward, batch_next_state):
        return self.loss_calculator(batch_state, batch_action, batch_reward, batch_next_state)


class NoLossBridge(AbstractLossBridge):
    def get(self, batch_state, batch_action, batch_reward, batch_next_state):
        return 0
