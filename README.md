Energizer (formerly `energizer`) – custom PyTorch-like library for the Neural Engine

## Need to Implement:

### A. Essential Layers:

- [x] Batch Normalization - Critical for stable training (BatchNorm1d, BatchNorm2d)
- [x] Dropout - For regularization
- [x] Pooling layers (MaxPool2d, AvgPool2d)
- [x] Flatten layer - To convert conv outputs to linear inputs
- [x] Residual blocks - For deeper networks

### B. Activation Functions:

- [x] Sigmoid - For binary classification
- [x] Tanh - For value estimation (-1 to 1)
- [x] LeakyReLU - Better than ReLU in some cases
- [ ] Softmax - For move probability distribution

### C. Loss Functions:

- [x] MSE Loss - For value prediction
- [ ] CrossEntropy Loss - For move prediction
- [ ] Huber Loss - More robust than MSE
- [ ] Custom chess loss - Combine policy and value loss
