from typing import Callable
from energizer.neural_network import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        **kwargs,
    ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )

        if isinstance(params, (list, tuple)) and len(params) > 0:
            if isinstance(params[0], dict):
                params = params
            else:
                params = [{"params": list(params)}]
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                p_data = p.data
                if hasattr(p_data, "__class__") and "mlx" in str(type(p_data)):
                    import mlx.core as mx

                    exp_avg = mx.zeros_like(p_data)
                    exp_avg_sq = mx.zeros_like(p_data)
                    max_exp_avg_sq = mx.zeros_like(p_data) if group["amsgrad"] else None
                else:
                    exp_avg = np.zeros_like(p_data)
                    exp_avg_sq = np.zeros_like(p_data)
                    max_exp_avg_sq = np.zeros_like(p_data) if group["amsgrad"] else None

                self.state[p] = {
                    "step": 0,
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                }
                if group["amsgrad"]:
                    self.state[p]["max_exp_avg_sq"] = max_exp_avg_sq

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
            group.setdefault("decoupled_decay", False)
            group.setdefault("foreach", False)
            group.setdefault("amsgrad", False)
            group.setdefault("weight_decay", 0)
            group.setdefault("beta1", 0.9)
            group.setdefault("beta2", 0.999)
            group.setdefault("eps", 1e-08)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data if hasattr(p.grad, "data") else p.grad

                if not isinstance(grad, np.ndarray):
                    grad = np.array(grad)

                p_data = p.data
                is_mlx = hasattr(p_data, "__class__") and "mlx" in str(type(p_data))
                if is_mlx and not (
                    hasattr(grad, "__class__") and "mlx" in str(type(grad))
                ):
                    import mlx.core as mx

                    grad = mx.array(grad)
                elif not is_mlx and (
                    hasattr(grad, "__class__") and "mlx" in str(type(grad))
                ):
                    grad = np.array(grad)

                if maximize:
                    grad = -grad

                if weight_decay != 0:
                    grad = grad + weight_decay * p_data

                state = self.state[p]
                state["step"] += 1

                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad
                state["exp_avg_sq"] = (
                    beta2 * state["exp_avg_sq"] + (1 - beta2) * grad * grad
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_hat = state["exp_avg"] / bias_correction1
                exp_avg_sq_hat = state["exp_avg_sq"] / bias_correction2

                if amsgrad:
                    if is_mlx:
                        import mlx.core as mx

                        state["max_exp_avg_sq"] = mx.maximum(
                            state["max_exp_avg_sq"], exp_avg_sq_hat
                        )
                        denom = mx.sqrt(state["max_exp_avg_sq"]) + eps
                    else:
                        state["max_exp_avg_sq"] = np.maximum(
                            state["max_exp_avg_sq"], exp_avg_sq_hat
                        )
                        denom = np.sqrt(state["max_exp_avg_sq"]) + eps
                else:
                    if is_mlx:
                        import mlx.core as mx

                        denom = mx.sqrt(exp_avg_sq_hat) + eps
                    else:
                        denom = np.sqrt(exp_avg_sq_hat) + eps

                p.data = p_data - lr * exp_avg_hat / denom

        return loss
