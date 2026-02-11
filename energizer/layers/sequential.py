from energizer.neural_network import Module
import energizer

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def forward(self, x: energizer.Tensor) -> energizer.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def to(self, device: str):
        for layer in self.layers:
            layer.to(device)
        return self