"""
Tests comparing energizer operators with PyTorch operators.
Tests mul, rmul, add, sub, and neg operations.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import energizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import energizer


class TestOperators(unittest.TestCase):
    """Test operators: mul, rmul, add, sub, and neg"""

    def setUp(self):
        """Set up test fixtures"""
        self.rtol = 1e-5  # Relative tolerance for floating point comparison
        self.atol = 1e-8  # Absolute tolerance for floating point comparison

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

    # ========== MUL TESTS ==========

    def test_mul_tensor_tensor_1d(self):
        """Test tensor * tensor for 1D tensors"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = energizer.tensor([4.0, 5.0, 6.0])
        result = a * b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = torch.tensor([4.0, 5.0, 6.0])
        result_torch = a_torch * b_torch
        result_torch.sum().backward()
        self.assert_tensors_close(a.grad, a_torch.grad, "1D tensor * tensor")

    def test_rmul_tensor_scalar(self):
        """Test scalar * tensor"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = 2 * a
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = 2 * a_torch
        result_torch.sum().backward()
        self.assert_tensors_close(a.grad, a_torch.grad, "scalar * tensor")

    # ========== ADD TESTS ==========

    def test_add_tensor_tensor_1d(self):
        """Test tensor + tensor for 1D tensors"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = energizer.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result = a + b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = a_torch + b_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "1D tensor + tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "1D tensor + tensor grad a")
        self.assert_tensors_close(b.grad, b_torch.grad, "1D tensor + tensor grad b")

    def test_add_tensor_scalar(self):
        """Test tensor + scalar"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = a + 2.0
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = a_torch + 2.0
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "tensor + scalar forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "tensor + scalar grad")

    def test_add_tensor_tensor_2d(self):
        """Test tensor + tensor for 2D tensors"""
        # energizer
        a = energizer.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = energizer.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result = a + b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result_torch = a_torch + b_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "2D tensor + tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "2D tensor + tensor grad a")
        self.assert_tensors_close(b.grad, b_torch.grad, "2D tensor + tensor grad b")

    # ========== SUB TESTS ==========

    def test_sub_tensor_tensor_1d(self):
        """Test tensor - tensor for 1D tensors"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = energizer.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result = a - b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = a_torch - b_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "1D tensor - tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "1D tensor - tensor grad a")
        self.assert_tensors_close(b.grad, b_torch.grad, "1D tensor - tensor grad b")

    def test_sub_tensor_scalar(self):
        """Test tensor - scalar"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = a - 2.0
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = a_torch - 2.0
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "tensor - scalar forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "tensor - scalar grad")

    def test_sub_tensor_tensor_2d(self):
        """Test tensor - tensor for 2D tensors"""
        # energizer
        a = energizer.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = energizer.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result = a - b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result_torch = a_torch - b_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "2D tensor - tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "2D tensor - tensor grad a")
        self.assert_tensors_close(b.grad, b_torch.grad, "2D tensor - tensor grad b")

    # ========== NEG TESTS ==========

    def test_neg_tensor_1d(self):
        """Test -tensor for 1D tensors"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = -a
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = -a_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "1D -tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "1D -tensor grad")

    def test_neg_tensor_2d(self):
        """Test -tensor for 2D tensors"""
        # energizer
        a = energizer.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result = -a
        result.backward()

        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result_torch = -a_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "2D -tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "2D -tensor grad")

    def test_neg_tensor_negative_values(self):
        """Test -tensor with negative values"""
        # energizer
        a = energizer.tensor([-1.0, -2.0, 3.0], requires_grad=True)
        result = -a
        result.backward()

        # PyTorch
        a_torch = torch.tensor([-1.0, -2.0, 3.0], requires_grad=True)
        result_torch = -a_torch
        result_torch.sum().backward()

        self.assert_tensors_close(
            result, result_torch, "-tensor with negative values forward"
        )
        self.assert_tensors_close(
            a.grad, a_torch.grad, "-tensor with negative values grad"
        )

    # ========== MUL ADDITIONAL TESTS ==========

    def test_mul_tensor_scalar(self):
        """Test tensor * scalar"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = a * 2.0
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = a_torch * 2.0
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "tensor * scalar forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "tensor * scalar grad")

    def test_mul_tensor_tensor_2d(self):
        """Test tensor * tensor for 2D tensors"""
        # energizer
        a = energizer.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = energizer.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result = a * b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result_torch = a_torch * b_torch
        result_torch.sum().backward()

        self.assert_tensors_close(result, result_torch, "2D tensor * tensor forward")
        self.assert_tensors_close(a.grad, a_torch.grad, "2D tensor * tensor grad a")
        self.assert_tensors_close(b.grad, b_torch.grad, "2D tensor * tensor grad b")

    # ========== TRUEDIV TESTS ==========
    def test_truediv_tensor_tensor_1d(self):
        """Test tensor / tensor for 1D tensors"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = energizer.tensor([4.0, 5.0, 6.0])
        result = a / b
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = torch.tensor([4.0, 5.0, 6.0])
        result_torch = a_torch / b_torch
        result_torch.sum().backward()
        self.assert_tensors_close(a.grad, a_torch.grad, "1D tensor / tensor grad a")

    def test_truediv_tensor_scalar(self):
        """Test tensor / scalar"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = a / 2.0
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = a_torch / 2.0
        result_torch.sum().backward()
        self.assert_tensors_close(a.grad, a_torch.grad, "tensor / scalar grad")

    # ========== RTRUEDIV TESTS ==========
    def test_rtruediv_scalar_tensor(self):
        """Test scalar / tensor"""
        # energizer
        a = energizer.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = 2.0 / a
        result.backward()

        # PyTorch
        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = 2.0 / a_torch
        result_torch.sum().backward()
        self.assert_tensors_close(a.grad, a_torch.grad, "scalar / tensor grad")


if __name__ == "__main__":
    unittest.main()
