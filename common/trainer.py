import math
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim

class Trainer(torch.nn.Module):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset = None,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        # Setup the training variables
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move model to device
        self.model.to(self.device)

        # Prepare optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Prepare data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = (
            DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
            if self.val_dataset is not None
            else None
        )

        # Lists to store metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_inference_time = None
        self.total_param_elements = self._count_parameter_elements()

    def forward(self, model_input: torch.Tensor):
        return self.model(model_input)

    def _count_parameter_elements(self) -> int:
        """Counts the total number of parameter elements in a model."""
        return sum(p.numel() for p in self.model.parameters())

    def train(
        self,
        epochs: int = 10,
        print_every: int = 1,
        plot_path: str = 'loss_plot.png',
        stats_path: str = 'inference_stats.txt'
    ):
        """Runs the training loop for a specified number of epochs."""
        # Put the model in training mode
        self.model.train()

        for epoch in range(1, epochs + 1):
            # Initialize the epoch parameters
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in self.train_loader:
                # Put the data to the specified device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Calculate the loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Call the backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Get the batch stats
                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            # Calculate the epoch stats
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{epochs} | "
                      f"Train Loss: {epoch_loss:.4f} | "
                      f"Train Acc: {epoch_acc:.4f}")

                # Validation
                if self.val_loader:
                    val_loss, val_acc = self.evaluate()
                    self.val_losses.append(val_loss)
                    self.val_accs.append(val_acc)

        # Save the loss and accuracy curve plots and inference data
        self._save_loss_plot(plot_path)
        self._save_inference_stats(stats_path)

    def evaluate(self):
        """Evaluates the model on the validation set (if provided)."""
        # Check if the validation loader is initialized
        assert self.val_loader is not None, "Validation loader is not initialized"

        # Put the model in evaluation mode and initialize the validation parameters
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        self.best_inference_time = (
            math.inf 
            if self.best_inference_time is None
            else self.best_inference_time
        )

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Put the target to the target device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Calculate the loss
                start_time = time.time()
                outputs = self.model(inputs)
                end_time = time.time()
                loss = self.criterion(outputs, targets)

                # Update the current best_tim
                curr_inference_time = end_time - start_time
                if curr_inference_time < self.best_inference_time:
                    self.best_inference_time = curr_inference_time

                # Calculate the batch stats
                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        # Calculate the evaluation stats
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct / total
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")
        return val_loss, val_acc

    def _save_loss_plot(self, save_path: str = 'loss_plot.png'):
        """Generates and saves the plot for training and validation losses."""

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train Loss')
        if self.val_loader:
            plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f'Loss plot saved to {save_path}')

    def _save_inference_stats(self, save_path: str = 'inference_stats.txt'):
        """Saves model parameter count and best inference time to a txt file."""
        with open(save_path, 'w') as f:
            f.write(f"Total Parameters: {self.total_params}\n")
            if self.inference_time is not None:
                f.write(f"Best Inference Time (s): {self.inference_time:.6f}\n")
            else:
                f.write("Inference time not measured.\n")
        print(f'Inference stats saved to {save_path}')

