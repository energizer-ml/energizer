"""
Tests comparing energizer Linear layer with PyTorch Linear layer.
Tests forward pass, backward pass, and various configurations.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import energizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import energizer


class TestLinear(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.rtol = 1e-4  # Relative tolerance for floating point comparison
        self.atol = (
            1e-6  # Absolute tolerance for floating point comparison (float32 precision)
        )
        np.random.seed(42)  # Set seed for reproducibility
        torch.manual_seed(42)

    def assert_tensors_close(self, my_tensor, torch_tensor, msg=""):
        """Helper to compare energizer tensor with PyTorch tensor"""
        # Convert PyTorch tensor to numpy if needed
        if isinstance(torch_tensor, torch.Tensor):
            torch_data = torch_tensor.detach().numpy()
        else:
            torch_data = np.array(torch_tensor)

        # Compare data
        np.testing.assert_allclose(
            my_tensor.data,
            torch_data,
            rtol=self.rtol,
            atol=self.atol,
            err_msg=f"Tensor data mismatch. {msg}",
        )

    def set_same_weights(self, my_layer, torch_layer):
        """Set the same weights and biases for both layers"""
        # Get PyTorch weights and biases
        weight_data = torch_layer.weight.data.detach().numpy()
        bias_data = (
            torch_layer.bias.data.detach().numpy()
            if torch_layer.bias is not None
            else None
        )

        # Set energizer weights and biases
        my_layer.weight.data = weight_data
        if bias_data is not None:
            my_layer.bias.data = bias_data

    # ========== FORWARD PASS TESTS ==========

    def test_linear_forward_basic(self):
        """Test basic forward pass: (128, 20) -> (128, 30)"""
        in_features, out_features = 20, 30
        batch_size = 128

        # Create layers
        my_layer = energizer.Linear(in_features, out_features)
        torch_layer = nn.Linear(in_features, out_features)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_features).astype(np.float32)
        my_input = energizer.tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(my_output, torch_output, "Linear forward pass")

    def test_linear_forward_bias_false(self):
        """Test forward pass with bias=False: (128, 20) -> (128, 30)"""
        in_features, out_features = 20, 30
        batch_size = 128

        # Create layers
        my_layer = energizer.Linear(in_features, out_features, bias=False)
        torch_layer = nn.Linear(in_features, out_features, bias=False)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_features).astype(np.float32)
        my_input = energizer.tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "Linear forward pass with bias=False"
        )


if __name__ == "__main__":
    unittest.main()
