from abc import abstractmethod
import torch
from typing import *


class Node:
    def __init__(self, in_prefix: str | List[str] = "", out_prefix: str = None):
        self.in_prefix = in_prefix
        first_in_prefix = in_prefix if not isinstance(in_prefix, list) else in_prefix[0]
        self.out_prefix = out_prefix if out_prefix is not None else first_in_prefix
        if len(self.out_prefix) > 0:
            if self.out_prefix[-1] == '+':
                self.out_prefix = self.out_prefix[:-1] + first_in_prefix
            elif self.out_prefix[0] == '+':
                self.out_prefix = first_in_prefix + self.out_prefix[1:]


    @abstractmethod
    def __call__(self, pipe, data: Dict[str, torch.Tensor]):
        pass