from .autograd import *

def tensor(data, requires_grad=False, device="cpu"):
    return Tensor(data, device=device, requires_grad=requires_grad)
