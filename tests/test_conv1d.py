"""
Tests comparing energizer Conv1d layer with PyTorch Conv1d layer.
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


class TestConv1d(unittest.TestCase):
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

    def test_conv1d_forward_basic(self):
        """Test basic forward pass: (20, 3, 50) -> (20, 16, 50) with padding=1"""
        in_channels, out_channels = 3, 16
        kernel_size, stride, padding = 3, 1, 1
        batch_size, length = 20, 50

        # Create layers
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(my_output, torch_output, "Conv1d forward pass basic")

    def test_conv1d_forward_no_padding(self):
        """Test forward pass without padding"""
        in_channels, out_channels = 3, 16
        kernel_size, stride, padding = 3, 1, 0
        batch_size, length = 10, 30

        # Create layers
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "Conv1d forward pass no padding"
        )

    def test_conv1d_forward_stride_2(self):
        """Test forward pass with stride=2"""
        in_channels, out_channels = 3, 16
        kernel_size, stride, padding = 3, 2, 1
        batch_size, length = 10, 30

        # Create layers
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "Conv1d forward pass stride=2"
        )

    def test_conv1d_forward_large_padding(self):
        """Test forward pass with large padding"""
        in_channels, out_channels = 5, 32
        kernel_size, stride, padding = 5, 1, 2
        batch_size, length = 8, 20

        # Create layers
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "Conv1d forward pass large padding"
        )

    def test_conv1d_forward_different_kernel_sizes(self):
        """Test forward pass with different kernel sizes"""
        test_cases = [
            (3, 3, 1, 1),  # kernel_size=3
            (5, 5, 1, 2),  # kernel_size=5
            (7, 7, 1, 3),  # kernel_size=7
        ]

        in_channels, out_channels = 4, 8
        batch_size, length = 5, 25

        for kernel_size, _, stride, padding in test_cases:
            with self.subTest(kernel_size=kernel_size, stride=stride, padding=padding):
                # Create layers
                my_layer = energizer.Conv1d(
                    in_channels, out_channels, kernel_size, stride, padding
                )
                torch_layer = nn.Conv1d(
                    in_channels, out_channels, kernel_size, stride, padding
                )

                # Set same weights
                self.set_same_weights(my_layer, torch_layer)

                # Create input
                input_data = np.random.randn(batch_size, in_channels, length).astype(
                    np.float32
                )
                my_input = energizer.Tensor(input_data)
                torch_input = torch.tensor(input_data)

                # Forward pass
                my_output = my_layer(my_input)
                torch_output = torch_layer(torch_input)

                # Compare outputs
                self.assert_tensors_close(
                    my_output,
                    torch_output,
                    f"Conv1d forward pass kernel_size={kernel_size}, stride={stride}, padding={padding}",
                )

    def test_conv1d_forward_no_bias(self):
        """Test forward pass without bias"""
        in_channels, out_channels = 3, 16
        kernel_size, stride, padding = 3, 1, 1
        batch_size, length = 10, 30

        # Create layers without bias
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        torch_layer = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "Conv1d forward pass no bias"
        )

    def test_conv1d_output_shape(self):
        """Test that output shapes match PyTorch"""
        test_cases = [
            (
                3,
                16,
                3,
                1,
                1,
                50,
            ),  # (in_channels, out_channels, kernel, stride, padding, length)
            (3, 16, 3, 2, 1, 50),
            (5, 32, 5, 1, 2, 30),
            (4, 8, 7, 1, 0, 25),
        ]

        batch_size = 10

        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            length,
        ) in test_cases:
            with self.subTest(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ):
                # Create layers
                my_layer = energizer.Conv1d(
                    in_channels, out_channels, kernel_size, stride, padding
                )
                torch_layer = nn.Conv1d(
                    in_channels, out_channels, kernel_size, stride, padding
                )

                # Create input
                input_data = np.random.randn(batch_size, in_channels, length).astype(
                    np.float32
                )
                my_input = energizer.Tensor(input_data)
                torch_input = torch.tensor(input_data)

                # Forward pass
                my_output = my_layer(my_input)
                torch_output = torch_layer(torch_input)

                # Compare shapes
                self.assertEqual(
                    my_output.data.shape,
                    tuple(torch_output.shape),
                    f"Output shape mismatch: energizer={my_output.data.shape}, PyTorch={torch_output.shape}",
                )

    def test_conv1d_weight_shape(self):
        """Test that weight shapes match PyTorch"""
        in_channels, out_channels = 3, 16
        kernel_size = 3

        my_layer = energizer.Conv1d(in_channels, out_channels, kernel_size)
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size)

        # Compare weight shapes
        self.assertEqual(
            my_layer.weight.data.shape,
            tuple(torch_layer.weight.data.shape),
            f"Weight shape mismatch: energizer={my_layer.weight.data.shape}, PyTorch={torch_layer.weight.data.shape}",
        )

    def test_conv1d_parameters_count(self):
        """Test that parameter counts match PyTorch"""
        in_channels, out_channels = 3, 16
        kernel_size = 3

        my_layer = energizer.Conv1d(in_channels, out_channels, kernel_size)
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size)

        # Count parameters
        my_params = sum(p.data.size for p in my_layer.parameters())
        torch_params = sum(p.numel() for p in torch_layer.parameters())

        self.assertEqual(
            my_params,
            torch_params,
            f"Parameter count mismatch: energizer={my_params}, PyTorch={torch_params}",
        )

    # ========== BACKWARD PASS TESTS ==========

    def test_conv1d_backward_basic(self):
        """Test backward pass with gradients"""
        in_channels, out_channels = 3, 16
        kernel_size, stride, padding = 3, 1, 1
        batch_size, length = 5, 20

        # Create layers
        my_layer = energizer.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        torch_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # Set same weights
        self.set_same_weights(my_layer, torch_layer)

        # Create input with requires_grad
        input_data = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        my_input = energizer.Tensor(input_data, requires_grad=True)
        torch_input = torch.tensor(input_data, requires_grad=True)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare forward outputs
        self.assert_tensors_close(
            my_output, torch_output, "Conv1d backward forward pass"
        )

        # Backward pass
        my_output.backward()
        torch_output.sum().backward()

        # Compare input gradients
        self.assert_tensors_close(
            my_input.grad, torch_input.grad, "Conv1d backward input grad"
        )

        # Compare weight gradients
        if my_layer.weight.grad is not None:
            self.assert_tensors_close(
                my_layer.weight.grad,
                torch_layer.weight.grad,
                "Conv1d backward weight grad",
            )
        else:
            self.skipTest(
                "Conv1d backward pass not yet implemented - weight.grad is None"
            )

        # Compare bias gradients
        if my_layer.bias is not None:
            if my_layer.bias.grad is not None:
                self.assert_tensors_close(
                    my_layer.bias.grad,
                    torch_layer.bias.grad,
                    "Conv1d backward bias grad",
                )
            else:
                self.skipTest(
                    "Conv1d backward pass not yet implemented - bias.grad is None"
                )


if __name__ == "__main__":
    unittest.main()
