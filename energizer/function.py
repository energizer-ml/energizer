from __future__ import annotations
from typing import Any, Callable, List


class Function:
    def __init__(self, function: Callable, tensors: List[Any], **kwargs: Any):
        self.function = function
        self.tensors = tensors
        self.kwargs = kwargs

    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return ctx.function(ctx.tensors, grad_outputs, **ctx.kwargs)
