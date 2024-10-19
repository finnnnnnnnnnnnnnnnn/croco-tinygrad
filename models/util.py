import itertools
from tinygrad.tensor import Tensor
import numpy as np

def cartesian_prod(*arrays):
    prod = list(itertools.product(*[arr.numpy().flatten() for arr in arrays]))
    return Tensor(prod)

def xavier_uniform(w):
    fan_in, fan_out = w.shape[0], np.prod(w.shape[1:])
    limit = np.sqrt(6 / (fan_in + fan_out))
    return Tensor(np.random.uniform(-limit, limit, size=w.shape))

def normal_init(tensor, std):
    tensor[:] = np.random.normal(0, std, size=tensor.shape)
