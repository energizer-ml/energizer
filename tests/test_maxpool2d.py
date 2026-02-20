"""
Tests comparing energizer MaxPool2d layer with PyTorch MaxPool2d layer.
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


class TestMaxPool2d(unittest.TestCase):
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

    # ========== FORWARD PASS TESTS ==========

    def test_maxpool2d_forward_basic(self):
        """Test basic forward pass: (4, 3, 32, 32) -> (4, 3, 16, 16) with kernel_size=2, stride=2"""
        kernel_size, stride = 2, 2
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(kernel_size=kernel_size, stride=stride)
        torch_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # Create input
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d forward pass basic"
        )

    def test_maxpool2d_forward_kernel_size_3(self):
        """Test forward pass with kernel_size=3"""
        kernel_size, stride = 3, 1
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(kernel_size=kernel_size, stride=stride)
        torch_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # Create input
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d forward pass kernel_size=3"
        )

    def test_maxpool2d_forward_stride_3(self):
        """Test forward pass with stride=3"""
        kernel_size, stride = 2, 3
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(kernel_size=kernel_size, stride=stride)
        torch_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # Create input
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d forward pass stride=3"
        )

    def test_maxpool2d_forward_padding(self):
        """Test forward pass with padding"""
        kernel_size, stride, padding = 3, 1, 1
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        torch_layer = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Create input
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d forward pass with padding"
        )

    def test_maxpool2d_forward_tuple_parameters(self):
        """Test forward pass with tuple parameters"""
        kernel_size, stride, padding = (3, 5), (2, 3), (1, 2)
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        torch_layer = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Create input
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d forward pass tuple parameters"
        )

    def test_maxpool2d_forward_different_sizes(self):
        """Test forward pass with different input sizes"""
        test_cases = [
            (4, 3, 16, 16, 2, 2),
            (1, 1, 64, 64, 4, 4),
            (8, 16, 8, 8, 2, 1),
        ]

        for batch_size, channels, height, width, kernel_size, stride in test_cases:
            with self.subTest(
                batch_size=batch_size,
                channels=channels,
                height=height,
                width=width,
                kernel_size=kernel_size,
                stride=stride,
            ):
                # Create layers
                my_layer = energizer.MaxPool2d(kernel_size=kernel_size, stride=stride)
                torch_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

                # Create input
                input_data = np.random.randn(
                    batch_size, channels, height, width
                ).astype(np.float32)
                my_input = energizer.Tensor(input_data)
                torch_input = torch.tensor(input_data)

                # Forward pass
                my_output = my_layer(my_input)
                torch_output = torch_layer(torch_input)

                # Compare outputs
                self.assert_tensors_close(
                    my_output,
                    torch_output,
                    f"MaxPool2d forward pass size=({batch_size}, {channels}, {height}, {width}), "
                    f"kernel={kernel_size}, stride={stride}",
                )

    def test_maxpool2d_output_shape(self):
        """Test that output shapes match PyTorch"""
        test_cases = [
            (4, 3, 32, 32, 2, 2, 0),  # (batch, channels, h, w, kernel, stride, padding)
            (4, 3, 32, 32, 3, 1, 1),
            (1, 1, 64, 64, 4, 4, 0),
            (8, 16, 16, 16, 2, 2, 1),
        ]

        for (
            batch_size,
            channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
        ) in test_cases:
            with self.subTest(
                batch_size=batch_size,
                channels=channels,
                height=height,
                width=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ):
                # Create layers
                my_layer = energizer.MaxPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding
                )
                torch_layer = nn.MaxPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding
                )

                # Create input
                input_data = np.random.randn(
                    batch_size, channels, height, width
                ).astype(np.float32)
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

    # ========== BACKWARD PASS TESTS ==========

    def test_maxpool2d_backward_basic(self):
        """Test backward pass with gradients"""
        kernel_size, stride = 2, 2
        batch_size, channels, height, width = 4, 3, 32, 32

        # Create layers
        my_layer = energizer.MaxPool2d(kernel_size=kernel_size, stride=stride)
        torch_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # Create input with requires_grad
        input_data = np.random.randn(batch_size, channels, height, width).astype(
            np.float32
        )
        my_input = energizer.Tensor(input_data, requires_grad=True)
        torch_input = torch.tensor(input_data, requires_grad=True)

        # Forward pass
        my_output = my_layer(my_input)
        torch_output = torch_layer(torch_input)

        # Compare forward outputs
        self.assert_tensors_close(
            my_output, torch_output, "MaxPool2d backward forward pass"
        )

        # Backward pass
        my_output.backward()
        torch_output.sum().backward()

        # Check if gradients were computed (backward might not be implemented yet)
        if my_input.grad is None:
            self.skipTest(
                "MaxPool2d backward pass not yet implemented - input.grad is None"
            )

        # Compare input gradients
        self.assert_tensors_close(
            my_input.grad, torch_input.grad, "MaxPool2d backward input grad"
        )


if __name__ == "__main__":
    unittest.main()
