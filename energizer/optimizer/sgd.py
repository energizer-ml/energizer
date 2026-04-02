from typing import Callable
import numpy as np
from energizer.neural_network import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        default = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "dampening": dampening,
            "nesterov": nesterov,
        }
        super().__init__(params, default)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("dampening", 0)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        self._synchronize_distributed_gradients()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data if hasattr(p.grad, "data") else p.grad

                p_data = p.data
                is_mlx = hasattr(p_data, "__class__") and "mlx" in str(type(p_data))

                if is_mlx:
                    try:
                        import mlx.core as mx

                        if not (
                            hasattr(grad, "__class__") and "mlx" in str(type(grad))
                        ):
                            grad = mx.array(np.array(grad))
                    except ImportError:
                        grad = np.array(grad)
                else:
                    if not isinstance(grad, np.ndarray):
                        grad = np.array(grad)

                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    if p not in self.state:
                        self.state[p] = {}
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        if is_mlx:
                            import mlx.core as mx

                            buf = param_state["momentum_buffer"] = mx.zeros_like(p_data)
                        else:
                            buf = param_state["momentum_buffer"] = np.zeros_like(p_data)
                    else:
                        buf = param_state["momentum_buffer"]
                    buf = momentum * buf + (1 - dampening) * grad
                    param_state["momentum_buffer"] = buf
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                p.data = p_data - lr * grad
        return loss
