import time
import numpy as np
import torch
from typing import *
from . import nodes
from .nodes import Node


class Compose:
    """
    Pipeline for Data Construction.
    """
    def __init__(self, nodes: List[Node] = []):
        self.nodes = nodes        
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index: int):
        return self.nodes[index]

    def __call__(self, data: Dict[str, torch.Tensor], timing: bool = False):
        for module in self.nodes:
            if timing:
                start = time.time()
            data = module(self, data)
            if timing:
                print(f"{module.__class__.__name__}: {time.time() - start:.3f} sec")
        return data

