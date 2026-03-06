"""
Energizer — A lightweight deep learning library for Apple's Neural Engine.

Energizer provides a PyTorch-like API for building, training, and running
neural networks with first-class support for Apple Silicon via the MLX backend.
It can also run on CPU using NumPy as a fallback, making it suitable for
prototyping and experimentation on any platform.

The library implements automatic differentiation, a modular layer system,
and GPU-accelerated tensor operations through a familiar interface.

Core Features:
    - **Tensor**: Multi-backend tensor with autograd support (CPU via NumPy, GPU via MLX).
    - **Module**: Base class for all neural network layers with parameter management,
      serialization (``save``/``load``), device transfer (``to``), and train/eval modes.
    - **Optimizer**: Base optimizer with ``step()``, ``zero_grad()``, and state management.

Tensor Operations:
    - Creation: ``tensor()``, ``Tensor.randn()``, ``Tensor.zeros()``, ``Tensor.ones()``
    - Arithmetic: ``+``, ``-``, ``*``, ``/``, ``@`` (matmul), ``**`` (power)
    - Reduction: ``sum()``, ``mean()``
    - Shape: ``reshape()``, ``view()``, ``T`` (transpose)
    - Device: ``cpu()``, ``gpu()``, ``to(device)``
    - Autograd: ``backward()`` for automatic gradient computation

Functional API (``energizer.functionnal``):
    - ``max(tensor, floor)`` — Element-wise maximum with a floor value.
    - ``as_strided(tensor, shape, strides)`` — Create a strided view of a tensor.
    - ``trace(tensor)`` — Compute the trace of a 2D matrix.

Optimizers:
    - ``SGD``  — Stochastic Gradient Descent with momentum, weight decay, and Nesterov support.
    - ``Adam`` — Adaptive Moment Estimation with bias correction and AMSGrad variant.

Layers:
    - **Linear**: ``Linear(in_features, out_features, bias=True)``
      Fully connected layer: ``y = x @ W^T + b``.

    - **Convolutional**:
        - ``Conv1d(in_channels, out_channels, kernel_size, stride, padding)``
        - ``Conv2d(in_channels, out_channels, kernel_size, stride, padding)``
        - ``ConvTranspose2d(in_channels, out_channels, kernel_size, ...)``
          Transposed (deconvolution) layer for upsampling.

    - **Activation Functions**:
        - ``ReLU()``           — Rectified Linear Unit: ``max(0, x)``.
        - ``LeakyReLU(slope)`` — Leaky variant: ``max(0, x) + slope * min(0, x)``.
        - ``Sigmoid()``        — Logistic sigmoid: ``1 / (1 + exp(-x))``.

    - **Normalization**:
        - ``BatchNorm1d(num_features)`` — Batch normalization over 2D/3D input.
        - ``BatchNorm2d(num_features)`` — Batch normalization over 4D input.

    - **Pooling**:
        - ``MaxPool2d(kernel_size, stride, padding)``
        - ``AvgPool2d(kernel_size, stride, padding)``

    - **Regularization**:
        - ``Dropout(p=0.5)`` — Randomly zeroes elements during training.

    - **Shape Manipulation**:
        - ``Flatten(start_dim, end_dim)`` — Flattens contiguous dimensions.
        - ``Reshape(shape)``              — Reshapes tensor to target shape.
        - ``Trim(start, end)``            — Trims spatial dimensions of 4D tensors.

    - **Composite / Containers**:
        - ``Sequential(*layers)``      — Chains layers in order.
        - ``ModuleList(modules)``      — List-like container for modules (no ``forward``).
        - ``ResidualBlock(channels)``   — Conv → BN → ReLU → Conv → BN + skip connection.
        - ``BottleneckBlock(in_ch, out_ch)`` — Bottleneck residual block (expansion=4).
        - ``AutoEncoder(device)``       — Pre-configured convolutional autoencoder.

Loss Functions:
    - ``MSELoss(reduction='mean')``          — Mean Squared Error loss.
    - ``CrossEntropyLoss(reduction='mean')`` — Cross-Entropy loss for classification.

Example::

    import energizer

    # Create a simple model
    model = energizer.Sequential(
        energizer.Linear(784, 128),
        energizer.ReLU(),
        energizer.Linear(128, 10),
    )

    # Move to GPU (Apple Neural Engine via MLX)
    model.to('gpu')

    # Forward pass
    x = energizer.Tensor.randn(32, 784, device='gpu')
    output = model(x)

    # Compute loss and backpropagate
    loss_fn = energizer.MSELoss()
    target = energizer.Tensor.zeros(32, 10, device='gpu')
    loss = loss_fn(output, target)
    loss.backward()

    # Optimize
    optimizer = energizer.Adam(model.parameters(), lr=0.001)
    optimizer.step()
    optimizer.zero_grad()
"""

__version_info__ = (0, 1, 5)
__version__ = ".".join(map(str, __version_info__))
__author__ = "Florian GRIMA"
__name__ = "energizer"
__description__ = "A lightweight deep learning library for Apple's Neural Engine."
__url__ = "https://github.com/fgrimaepitech/energizer"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Florian GRIMA"
__maintainer__ = "Florian GRIMA"
__status__ = "Development"


# ---------------------------------------------------------------------------
# Context Managers
# ---------------------------------------------------------------------------
class _NoGrad:
    """
    Simple context manager to disable gradient tracking.

    This is a lightweight analogue of torch.no_grad(). It temporarily sets
    `requires_grad` to False on all existing tensors that currently require
    gradients, and restores their original flags on exit.
    """

    def __enter__(self):
        # Track tensors whose requires_grad flag we flip.
        self._modified = []
        # Import here to avoid circular imports at module import time.
        try:
            import gc
        except ImportError:
            return self

        for obj in gc.get_objects():
            try:
                if isinstance(obj, Tensor) and getattr(obj, "requires_grad", False):
                    self._modified.append(obj)
                    obj.requires_grad = False
            except ReferenceError:
                continue
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in getattr(self, "_modified", []):
            t.requires_grad = True
        return False


def no_grad():
    """
    Context manager that disables autograd on existing tensors.

    Usage:

        import energizer as ez

        with ez.no_grad():
            out = model(x)
    """

    return _NoGrad()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from .tensor import Tensor
from .neural_network import Module, Optimizer

# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
from .functionnal import max, as_strided, trace

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
from .optimizer.sgd import SGD
from .optimizer.adam import Adam

# ---------------------------------------------------------------------------
# Layers — Linear & Convolutional
# ---------------------------------------------------------------------------
from .layers.linear import Linear
from .layers.conv import ConvNd, Conv1d, Conv2d
from .layers.convtranspose import ConvTranspose2d

# ---------------------------------------------------------------------------
# Layers — Activation Functions
# ---------------------------------------------------------------------------
from .layers.relu import ReLU, LeakyReLU
from .layers.sigmoid import Sigmoid
from .layers.gelu import GELU
from .layers.tanh import Tanh

# ---------------------------------------------------------------------------
# Layers — Normalization
# ---------------------------------------------------------------------------
from .layers.batch_norm import BatchNorm1d, BatchNorm2d
from .layers.layer_norm import LayerNorm

# ---------------------------------------------------------------------------
# Layers — Pooling
# ---------------------------------------------------------------------------
from .layers.pool import MaxPool2d, AvgPool2d

# ---------------------------------------------------------------------------
# Layers — Regularization
# ---------------------------------------------------------------------------
from .layers.dropout import Dropout

# ---------------------------------------------------------------------------
# Layers — Shape Manipulation
# ---------------------------------------------------------------------------
from .layers.flatten import Flatten
from .layers.reshape import Reshape
from .layers.trim import Trim

# ---------------------------------------------------------------------------
# Layers — Composite / Containers
# ---------------------------------------------------------------------------
from .layers.sequential import Sequential
from .layers.modulelist import ModuleList
from .layers.residual import ResidualBlock, BottleneckBlock
from .layers.autoencoder import AutoEncoder

# ---------------------------------------------------------------------------
# Layers — Transformer
# ---------------------------------------------------------------------------
from .layers.transformer import TransformerEncoderLayer, TransformerEncoder

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
from .layers.loss import MSELoss, CrossEntropyLoss

# ---------------------------------------------------------------------------
# Layers — Embedding
# ---------------------------------------------------------------------------
from .layers.embedding import Embedding

# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------
from .layers.init import zeros_, orthogonal_
