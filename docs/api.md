# Energizer API reference

Devices: `"cpu"` (NumPy) and `"mlx"` (MLX on the Apple GPU through Metal). `"gpu"` is a deprecated alias of `"mlx"`.

### Tensor

The core data structure. Wraps NumPy arrays (`cpu`) and MLX arrays (`mlx`, Apple GPU via Metal).

```python
t = energizer.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Creation helpers
energizer.Tensor.randn(3, 4)
energizer.Tensor.zeros((3, 4))
energizer.Tensor.ones((3, 4))

# Device transfer
t.to("mlx")   # → Apple GPU via MLX/Metal
t.to("cpu")   # → NumPy

# Supported operators
t + t  |  t - t  |  t * t  |  t / t
t @ t  |  t ** 2 |  -t
t.sum()  |  t.mean()  |  t.T
t.reshape((4, 2))  |  t.view((4, 2))
t.transpose(0, 1)

# Autograd
loss = (model(x) - target).mean()
loss.backward()
```

---

### Module

Base class for all layers. Subclass it to define custom layers.

```python
class MyLayer(energizer.Module):
    def __init__(self):
        super().__init__()
        self.w = energizer.Parameter(energizer.Tensor.randn(4, 4))

    def forward(self, x):
        return x @ self.w

model.parameters()        # list of trainable Parameters
model.to("mlx")           # move all parameters to the MLX backend
model.train() / .eval()   # toggle training mode (affects Dropout, BatchNorm)
model.save("model.npz")   # serialize to disk
model.load("model.npz")   # restore from disk
```

---

### Layers

#### Linear

```python
energizer.Linear(in_features=128, out_features=64, bias=True)
```

#### Convolutional

```python
energizer.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
energizer.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
energizer.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

#### Activation Functions

```python
energizer.ReLU()
energizer.LeakyReLU(negative_slope=0.01)
energizer.Sigmoid()
energizer.GELU()
```

#### Normalization

```python
energizer.BatchNorm1d(num_features)
energizer.BatchNorm2d(num_features)
energizer.LayerNorm(normalized_shape)
```

#### Pooling

```python
energizer.MaxPool2d(kernel_size, stride=None, padding=0)
energizer.AvgPool2d(kernel_size, stride=None, padding=0)
```

#### Regularization

```python
energizer.Dropout(p=0.5)
```

#### Shape Manipulation

```python
energizer.Flatten(start_dim=1, end_dim=-1)
energizer.Reshape(shape)
energizer.Trim(start, end)
```

#### Containers

```python
energizer.Sequential(*layers)          # forward through layers in order
energizer.ModuleList([layer1, layer2]) # list of modules, no auto-forward
```

#### Residual Blocks

```python
energizer.ResidualBlock(channels)
energizer.BottleneckBlock(in_channels, out_channels)
```

#### Transformer

```python
energizer.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)
energizer.TransformerEncoder(encoder_layer, num_layers)
```

#### Embedding

```python
energizer.Embedding(num_embeddings, embedding_dim)
```

#### AutoEncoder

```python
energizer.AutoEncoder(device="cpu")   # pre-configured convolutional autoencoder
```

---

### Loss Functions

```python
energizer.MSELoss(reduction="mean")
energizer.CrossEntropyLoss(reduction="mean")
```

---

### Optimizers

#### SGD

```python
energizer.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=False,
)
```

#### Adam

```python
energizer.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
)
```

---

### Functional API

```python
from energizer import functionnal as F

F.max(tensor, floor=0.0)                     # element-wise max with a floor
F.as_strided(tensor, shape, strides)         # strided view of a tensor
F.trace(tensor)                              # trace of a 2D matrix
```

---

## Training Loop Example

```python
import energizer

model     = energizer.Sequential(energizer.Linear(4, 8), energizer.ReLU(), energizer.Linear(8, 1))
optimizer = energizer.Adam(model.parameters(), lr=1e-3)
loss_fn   = energizer.MSELoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()

    x      = energizer.Tensor.randn(16, 4)
    target = energizer.Tensor.zeros((16, 1))

    output = model(x)
    loss   = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
```

---

