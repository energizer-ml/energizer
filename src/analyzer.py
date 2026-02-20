import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import energizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
RANDOM_SEED = 49
LEARNING_RATE = 0.0005
BATCH_SIZE = 256  # Increased for GPU (128 was too small)
NUM_EPOCHS = 30  # Full training
NUM_CLASSES = 10


def get_device():
    """Get the best available device."""
    try:
        import mlx.core as mx

        # Check if MLX is available (Apple Silicon GPU)
        test_tensor = mx.zeros(1)
        print("✅ MLX GPU available")
        return "gpu"
    except ImportError:
        print("❌ MLX not available")
    except Exception as e:
        print(f"❌ MLX error: {e}")

    print("✅ Using CPU")
    return "cpu"


def get_dataloaders_mnist(batch_size: int, num_workers: int = 2):
    """Get MNIST dataloaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )

    test_dataset = datasets.MNIST(root="data", train=False, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,  # Faster data transfer to GPU
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, test_loader


def mse_loss(pred, target):
    """Efficient MSE loss that works on GPU."""
    diff = pred - target
    return (diff * diff).mean()


class AutoEncoderTrainer:
    """Helper class to manage GPU training."""

    def __init__(self, device="gpu"):
        self.device = device
        self.model = None
        self.optimizer = None

    def setup(self):
        """Setup model and optimizer on correct device."""
        print(f"\n🔧 Setting up on {self.device.upper()}")

        # Create model with device parameter
        self.model = energizer.AutoEncoder(device=self.device)

        # Move all parameters to device
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.data.size for p in self.model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

        # Create optimizer
        self.optimizer = energizer.Adam(self.model.parameters(), lr=LEARNING_RATE)

        return self.model, self.optimizer

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (features, _) in enumerate(train_loader):
            # Convert to numpy and create tensor on correct device
            features_np = features.numpy()

            # Create tensor directly on GPU if possible
            features_tensor = energizer.tensor(
                features_np, requires_grad=True, device=self.device
            )

            # Forward pass
            output = self.model(features_tensor)

            # Loss
            loss = mse_loss(output, features_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            loss_value = (
                float(loss.data.item())
                if hasattr(loss.data, "item")
                else float(loss.data)
            )
            total_loss += loss_value
            batch_count += 1

            # Logging
            if batch_idx % 50 == 0:
                print(
                    f"  Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss_value:.4f}"
                )

        return total_loss / batch_count if batch_count > 0 else 0.0

    def evaluate(self, test_loader, num_samples=5):
        """Evaluate model and visualize results."""
        self.model.eval()

        print(f"\n📊 Evaluating on {self.device.upper()}...")

        # Get test samples
        test_batch = next(iter(test_loader))[0]
        test_np = test_batch.numpy()[:num_samples]
        test_tensor = energizer.tensor(test_np, requires_grad=False, device=self.device)

        # Reconstruct
        with torch.no_grad():  # Disable PyTorch grad, but energizer might have its own
            reconstructed = self.model(test_tensor)

        # Convert to numpy for visualization
        original_np = test_tensor.numpy()
        reconstructed_np = reconstructed.numpy()

        # Visualize
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 12))

        for i in range(num_samples):
            # Original
            original_img = original_np[i, 0]  # Remove channel dimension
            axes[i, 0].imshow(original_img, cmap="gray")
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")

            # Reconstructed
            recon_img = reconstructed_np[i, 0]
            axes[i, 1].imshow(recon_img, cmap="gray")
            axes[i, 1].set_title(f"Reconstructed {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

        # Calculate test loss
        test_loss = 0.0
        batch_count = 0

        for data, _ in test_loader:
            if batch_count >= 10:  # Limit evaluation
                break

            data_np = data.numpy()
            data_tensor = energizer.tensor(
                data_np, requires_grad=False, device=self.device
            )

            output = self.model(data_tensor)
            loss = mse_loss(output, data_tensor)

            test_loss += (
                float(loss.data.item())
                if hasattr(loss.data, "item")
                else float(loss.data)
            )
            batch_count += 1

        avg_loss = test_loss / batch_count
        print(f"✅ Test Loss: {avg_loss:.4f}")

        return avg_loss


def train_autoencoder(train_loader, test_loader, device="gpu", num_epochs=30):
    """Main training function optimized for GPU."""

    print("=" * 60)
    print("🚀 STARTING AUTOENCODER TRAINING")
    print(f"📱 Device: {device.upper()}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"📈 Learning rate: {LEARNING_RATE}")
    print(f"⏰ Epochs: {num_epochs}")
    print("=" * 60)

    # Setup trainer
    trainer = AutoEncoderTrainer(device=device)
    model, optimizer = trainer.setup()

    # Training history
    history = {"train_loss": [], "test_loss": [], "epoch_times": []}

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"\n🎯 Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        history["train_loss"].append(train_loss)

        # Evaluate
        test_loss = trainer.evaluate(test_loader)
        history["test_loss"].append(test_loss)

        # Timing
        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(epoch_time)

        print(f"📊 Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"⏱️  Epoch time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, f"checkpoint_epoch_{epoch+1}.npz"
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes")
    print(f"📈 Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"📉 Final Test Loss: {history['test_loss'][-1]:.4f}")
    print("=" * 60)

    # Plot training history
    plot_training_history(history)

    return model, history


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    # Convert tensors to numpy for saving
    for key in ["model_state", "optimizer_state"]:
        if key in checkpoint:
            # Convert energizer tensors to numpy
            state_dict = checkpoint[key]
            numpy_state = {}
            for k, v in state_dict.items():
                if hasattr(v, "numpy"):
                    numpy_state[k] = v.numpy()
                else:
                    numpy_state[k] = v
            checkpoint[key] = numpy_state

    np.savez(filename, **checkpoint)
    print(f"💾 Checkpoint saved: {filename}")


def plot_training_history(history):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["test_loss"], label="Test Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Time plot
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch_times"], "o-", label="Epoch Time", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Training Time per Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def execute_train(args):
    """Main execution function."""

    # Get best device
    device = get_device()

    print("\n📥 Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders_mnist(BATCH_SIZE)

    print(f"📊 Training samples: {len(train_loader.dataset):,}")
    print(f"📊 Test samples: {len(test_loader.dataset):,}")
    print(f"📊 Batches per epoch: {len(train_loader)}")

    # Train
    model, history = train_autoencoder(
        train_loader, test_loader, device=device, num_epochs=NUM_EPOCHS
    )

    # Save final model
    final_filename = "autoencoder_final.npz"
    save_checkpoint(model, None, NUM_EPOCHS, history["test_loss"][-1], final_filename)

    print(f"\n💾 Model saved to: {final_filename}")
    print("🎉 Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(prog="energizer_analyzer")
    exec_mode = parser.add_mutually_exclusive_group(required=True)
    exec_mode.add_argument("--train", action="store_true", help="Train the autoencoder")
    exec_mode.add_argument(
        "--eval", action="store_true", help="Evaluate a trained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="autoencoder_final.npz",
        help="Model file to load/save",
    )

    args = parser.parse_args()

    if args.train:
        execute_train(args)
    elif args.eval:
        print("Evaluation mode not implemented yet")
        # You can add evaluation code here


if __name__ == "__main__":
    main()
