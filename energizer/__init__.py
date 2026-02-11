__version__ = "0.1.1"
__author__ = "Florian GRIMA"
__name__ = "energizer"
__description__ = "A lightweight deep learning library for Apple's Neural Engine."
__url__ = "https://github.com/fgrimaepitech/energizer"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Florian GRIMA"
__version_info__ = (0, 1, 1)
__version__ = ".".join(map(str, __version_info__))
__maintainer__ = "Florian GRIMA"
__status__ = "Development"

# Core
from .tensor import Tensor, tensor
from .neural_network import Module, Optimizer

# Functionnal
from .functionnal import max, as_strided, trace

# Optimizer
from .optimizer.sgd import SGD
from .optimizer.adam import Adam

# Layers
from .layers.linear import Linear
from .layers.relu import ReLU, LeakyReLU
from .layers.conv import ConvNd, Conv1d, Conv2d
from .layers.batch_norm import BatchNorm1d, BatchNorm2d
from .layers.dropout import Dropout
from .layers.pool import MaxPool2d, AvgPool2d
from .layers.flatten import Flatten
from .layers.residual import ResidualBlock, BottleneckBlock
from .layers.sequential import Sequential
from .layers.autoencoder import AutoEncoder
from .layers.convtranspose import ConvTranspose2d
from .layers.reshape import Reshape
from .layers.trim import Trim
from .layers.sigmoid import Sigmoid
from .layers.loss import MSELoss