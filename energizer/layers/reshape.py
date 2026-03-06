from energizer.neural_network import Module
from energizer.tensor import Tensor
import energizer.autograd as autograd


class Reshape(Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor, got {type(x)}")

        new_shape = list(self.shape)

        if -1 in new_shape:
            total_elements = np.prod(x.shape)
            known_elements = np.prod([abs(d) for d in new_shape if d != -1])
            inferred_dim = total_elements // known_elements
            new_shape[new_shape.index(-1)] = inferred_dim

        new_shape = tuple(new_shape)

        return autograd.Reshape.apply(x, new_shape)
