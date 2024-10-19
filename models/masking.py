# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Masking utils
# --------------------------------------------------------

# import tinygrad.nn as nn
import tinygrad
import numpy as np
    
class RandomMask:
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = tinygrad.Tensor.rand(x.size(0), self.num_patches, device=x.device) # -> replace torch.rand with tinygrad.Tensor.rand
        argsort = tinygrad.Tensor(np.argsort(noise.numpy(), axis=1)) # -> replaced torch.argsort with tinygrad.helpers.argsort
        return argsort < self.num_mask