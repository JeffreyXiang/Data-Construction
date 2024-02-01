from abc import abstractmethod
import time
import numpy as np
import torch
from typing import *
import inspect
from . import nodes
from .nodes import Node


__all__ = [
    'Pipeline',
    'Compose',
    'Node'
    'nodes'
]


class Pipeline:
    """
    Pipeline for Data Construction.
    """
    def __init__(self, nodes: List[Node] = []):
        self.nodes = nodes
        self.registered_components = {}
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index: int):
        return self.nodes[index]

    def get_shared_component(self, name: str, init_fn: Callable, init_args: tuple = (), init_kwargs: dict = {}):
        """
        Get shared model.

        Args:
            name (str): Name of the component.
            init_fn (Callable): Function to initialize the component.
        """
        if name not in self.registered_components:
            self.registered_components[name] = {
                'init_fn': init_fn,
                'init_args': init_args,
                'init_kwargs': init_kwargs,
                'component': init_fn(*init_args, **init_kwargs)
            }

        # Check if the component has been registered without conflict.
        assert self.registered_components[name]['init_fn'] == init_fn, f"Component {name} has been registered with different init_fn."
        signature1 = inspect.signature(init_fn)
        bound_args1 = signature1.bind(*init_args, **init_kwargs)
        bound_args1.apply_defaults()
        signature2 = inspect.signature(self.registered_components[name]['init_fn'])
        bound_args2 = signature2.bind(*self.registered_components[name]['init_args'], **self.registered_components[name]['init_kwargs'])
        bound_args2.apply_defaults()
        for k in signature1.parameters:
            assert bound_args1.arguments[k] == bound_args2.arguments[k], f"Component {name} has been registered with different init_args or init_kwargs."
        
        return self.registered_components[name]['component']

    @abstractmethod
    def __call__(self, data: Dict[str, torch.Tensor], timing: bool = False):
        pass


class Compose(Pipeline):
    """
    Pipeline for Data Construction.
    """
    def __init__(self, nodes: List[Node] = []):
        super().__init__(nodes)

    def __call__(self, data: Dict[str, torch.Tensor], timing: bool = False):
        for module in self.nodes:
            if timing:
                start = time.time()
            data = module(data, pipe=self)
            if timing:
                print(f"{module.__class__.__name__}: {time.time() - start:.3f} sec")
        return data

