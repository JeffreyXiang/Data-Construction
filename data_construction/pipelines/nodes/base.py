from abc import abstractmethod
import torch
from typing import *


class Node:
    def __init__(self, in_prefix: Union[str, List[str]] = "", out_prefix: str = None):
        self.in_prefix = in_prefix
        first_in_prefix = in_prefix if not isinstance(in_prefix, list) else in_prefix[0]
        self.out_prefix = out_prefix if out_prefix is not None else first_in_prefix
        if len(self.out_prefix) > 0:
            if self.out_prefix[-1] == '+':
                self.out_prefix = self.out_prefix[:-1] + first_in_prefix
            elif self.out_prefix[0] == '+':
                self.out_prefix = first_in_prefix + self.out_prefix[1:]

    def get_lazy_component(self, name: str, init_fn: Callable, init_args: tuple = (), init_kwargs: dict = {}, pipe=None):
        """
        Lazy initialization of component.
        If the component has not been initialized, initialize it.
        If the component has been initialized, return it.
        If the pipeline is provided, the component will be managed by the pipeline in a shared manner.

        Args:
            name (str): Name of the component.
            pipe (Pipeline): Pipeline.
            init_fn (Callable): Function to initialize the component.
        """
        if not hasattr(self, name):
            if pipe is not None:
                setattr(self, name, pipe.get_shared_component(name, init_fn, init_args, init_kwargs))
            else:
                setattr(self, name, init_fn(*init_args, **init_kwargs))
        return getattr(self, name)

    @abstractmethod
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        pass