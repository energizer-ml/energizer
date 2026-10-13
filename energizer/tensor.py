from energizer.backend import normalize_device
from .autograd import *


def tensor(data, requires_grad=False, device="cpu"):
    device = normalize_device(device)
    return Tensor(data, device=device, requires_grad=requires_grad)
