"""
Tests comparing energizer as_strided with PyTorch as_strided.
Tests various shapes, strides, and storage_offset configurations.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import energizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import energizer


class TestAsStrided(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.rtol = 1e-5  # Relative tolerance for floating point comparison
        self.atol = 1e-8  # Absolute tolerance for floating point comparison
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

    def assert_tensors_equal(self, my_tensor, torch_tensor, msg=""):
        """Helper to compare energizer tensor with PyTorch tensor (exact match for integers)"""
        # Convert PyTorch tensor to numpy if needed
        if isinstance(torch_tensor, torch.Tensor):
            torch_data = torch_tensor.detach().numpy()
        else:
            torch_data = np.array(torch_tensor)

        # Compare data exactly
        np.testing.assert_array_equal(
            my_tensor.data, torch_data, err_msg=f"Tensor data mismatch. {msg}"
        )

    # ========== BASIC TESTS ==========

    def test_as_strided_basic_2x2(self):
        """Test basic as_strided: (2, 3) -> (2, 2) with strides (1, 2)"""
        # Create input tensors
        input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided
        shape = (2, 2)
        strides = (1, 2)
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "Basic as_strided (2, 2) with strides (1, 2)"
        )

    def test_as_strided_basic_3x3(self):
        """Test as_strided: (3, 3) -> (2, 2) with different strides"""
        # Create input tensors
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided
        shape = (2, 2)
        strides = (3, 1)  # Skip rows, step through columns
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided (2, 2) with strides (3, 1)"
        )

    # ========== DIFFERENT SHAPES ==========

    def test_as_strided_shape_1d(self):
        """Test as_strided with 1D output shape"""
        # Create input tensors
        input_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided to get every other element
        shape = (3,)
        strides = (2,)
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "1D as_strided with stride 2"
        )

    def test_as_strided_shape_3d(self):
        """Test as_strided with 3D output shape"""
        # Create input tensors
        input_data = np.arange(24, dtype=np.int64).reshape(4, 3, 2)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided
        shape = (2, 2, 2)
        strides = (6, 2, 1)  # Stride through the 3D array
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(my_output, torch_output, "3D as_strided")

    def test_as_strided_shape_larger(self):
        """Test as_strided with output shape that repeats elements (valid bounds)"""
        # Create input tensors with enough data
        input_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided to create a view that repeats elements but stays in bounds
        # This creates a view that reads the same row twice
        shape = (2, 2)
        strides = (0, 1)  # Zero stride repeats the first row
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided with zero stride repetition"
        )

    # ========== DIFFERENT STRIDES ==========

    def test_as_strided_strides_large(self):
        """Test as_strided with large strides (skipping elements)"""
        # Create input tensors
        input_data = np.arange(20, dtype=np.int64).reshape(4, 5)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided with large strides
        shape = (2, 2)
        strides = (10, 2)  # Skip rows and columns
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided with large strides"
        )

    def test_as_strided_strides_zero(self):
        """Test as_strided with zero stride (broadcasting)"""
        # Create input tensors
        input_data = np.array([1, 2, 3], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided with zero stride to repeat elements
        shape = (3, 3)
        strides = (0, 1)  # Zero stride repeats rows
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided with zero stride"
        )

    # ========== STORAGE OFFSET TESTS ==========

    def test_as_strided_storage_offset(self):
        """Test as_strided with storage_offset"""
        # Create input tensors
        input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided with storage_offset
        shape = (2, 2)
        strides = (3, 1)
        storage_offset = 1  # Start from index 1
        my_output = energizer.as_strided(my_input, shape, strides, storage_offset)
        torch_output = torch.as_strided(torch_input, shape, strides, storage_offset)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided with storage_offset"
        )

    def test_as_strided_storage_offset_2d(self):
        """Test as_strided with storage_offset on 2D input"""
        # Create input tensors
        input_data = np.arange(12, dtype=np.int64).reshape(3, 4)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided with storage_offset
        shape = (2, 2)
        strides = (4, 1)
        storage_offset = 2  # Start from element at flat index 2
        my_output = energizer.as_strided(my_input, shape, strides, storage_offset)
        torch_output = torch.as_strided(torch_input, shape, strides, storage_offset)

        # Compare outputs
        self.assert_tensors_equal(
            my_output, torch_output, "as_strided with storage_offset on 2D"
        )

    # ========== FLOATING POINT TESTS ==========

    def test_as_strided_float(self):
        """Test as_strided with floating point data"""
        # Create input tensors
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided
        shape = (2, 2)
        strides = (1, 2)
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_close(my_output, torch_output, "as_strided with float data")

    def test_as_strided_random_float(self):
        """Test as_strided with random floating point data"""
        # Create input tensors
        input_data = np.random.randn(4, 5).astype(np.float32)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided
        shape = (3, 3)
        strides = (5, 1)
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_close(
            my_output, torch_output, "as_strided with random float data"
        )

    # ========== EDGE CASES ==========

    def test_as_strided_single_element(self):
        """Test as_strided returning a single element"""
        # Create input tensors
        input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided to get single element
        shape = (1, 1)
        strides = (3, 1)
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(my_output, torch_output, "as_strided single element")

    def test_as_strided_same_shape(self):
        """Test as_strided with same shape as input"""
        # Create input tensors
        input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided with same shape
        shape = (2, 3)
        strides = (3, 1)  # Normal row-major strides
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(my_output, torch_output, "as_strided same shape")

    def test_as_strided_transpose_view(self):
        """Test as_strided to create transpose-like view"""
        # Create input tensors
        input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        my_input = energizer.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        # Apply as_strided to transpose (swap strides)
        shape = (3, 2)
        strides = (1, 2)  # Transposed strides
        my_output = energizer.as_strided(my_input, shape, strides)
        torch_output = torch.as_strided(torch_input, shape, strides)

        # Compare outputs
        self.assert_tensors_equal(my_output, torch_output, "as_strided transpose view")


if __name__ == "__main__":
    unittest.main()
