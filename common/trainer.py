import logging
import math
import os
import time
import torch
import matplotlib.pyplot as plt
from argparse import Namespace
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim

class Trainer(torch.nn.Module):
    def __init__(
        self,
        model: nn.Module,
        arguments: Namespace,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        optimizer: optim.Optimizer | None = None,
        criterion: torch.nn.Module | None = None,
    ):
        super().__init__()

        # Setup the training variables
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = arguments.batch_size
        self.device = arguments.device if arguments.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = arguments.patience
        self.lr_schedule = arguments.lr_schedule
        self.num_classes = arguments.num_classes

        # Move model to device
        self.model.to(self.device)

        # Prepare optimizer and loss
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.SGD(
                self.model.parameters(),
                lr=arguments.lr,
                momentum=arguments.momentum, 
                weight_decay=arguments.weight_decay,
            )
        )
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Setup lr scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_schedule, gamma=arguments.gamma)

        # Prepare data loaders
        self.train_loader = (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=arguments.num_workers,
            )
            if self.train_dataset is not None
            else None
        )
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=arguments.num_workers,
            )
            if self.val_dataset is not None
            else None
        )

        # Handle the storage folder
        timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.output_path = arguments.output_path + "/" + timestamp + "/"
        os.makedirs(self.output_path, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        log_path = self.output_path + "run.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Lists to store metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_consistencies = []
        self.best_inference_time = None
        self.total_param_elements = self._count_parameter_elements()

        # Consistancy variables
        self.num_parts = arguments.num_parts
        self.consistency_treshold = arguments.mu
        self.image_size = arguments.image_size
        self.box_size = arguments.box_size

        # storage for validation data
        self.validation_activation_maps = []
        self.validation_target_labels = []
        self.validation_part_annotations = []

    def forward(self, model_input: torch.Tensor):
        return self.model(model_input)

    def _count_parameter_elements(self) -> int:
        """Counts the total number of parameter elements in a model."""
        return sum(p.numel() for p in self.model.parameters())

    def train(
        self,
        epochs: int = 10,
        print_every: int = 1,
    ):
        """Runs the training loop for a specified number of epochs."""
        self.logger.info("Starting the training")

        # Clear the loss and accuracy arrays
        self.train_accs, self.train_losses = [], []
        self.val_accs, self.val_losses = [], []

        # Check if the training loader is initialized
        assert self.train_loader is not None, "Training loader is not initialized"

        best_val_acc = -1
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # Put the model in training mode
            self.model.train()

            # Initialize the epoch parameters
            running_loss = 0.0
            correct = 0
            total = 0

            for idx, batch in enumerate(self.train_loader):
                if idx % print_every == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} | Training step: {idx}/{len(self.train_loader)} | {(idx/len(self.train_loader) * 100):.2f} %"
                    )

                # Check for partial maps
                keypoints = None
                if len(batch) == 3:
                    inputs, targets, keypoints = batch
                else:
                    inputs, targets = batch

                # Put the data to the specified device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Call the training step
                step_loss, step_correct = self.train_step(inputs, targets, epoch=epoch, keypoints=keypoints)

                # Update the train metrics
                running_loss += step_loss
                correct += step_correct
                total += targets.size(0)

            # Calculate the epoch stats
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            self.logger.info(
                f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}"
            )

            # Validation
            if self.val_loader:
                self.evaluate(print_every=print_every)

                if self.val_accs[-1] > best_val_acc:
                    best_val_acc = self.val_accs[-1]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"No improvement in {self.patience} epochs." )
                        break

            # Call the scheduler step
            self.scheduler.step()

        # Save the loss and accuracy curve plots and inference data
        self._save_plots()
        self._save_inference_stats()
        self._save_checkpoint()

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> tuple[float, float]:
        # Zero the gradients
        self.optimizer.zero_grad()

        # Calculate the loss
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Call the backward pass
        loss.backward()
        self.optimizer.step()

        # Get the batch stats
        running_loss = loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()

        return running_loss, correct

    def evaluate(self, print_every = 100):
        """Evaluates the model on the validation set (if provided)."""
        self.logger.info("Starting the validation")

        # Clear consistancy variables
        self.validation_activation_maps.clear()
        self.validation_target_labels.clear()
        self.validation_part_annotations.clear()

        # If evaluate is called on its own, clear the loss and accuracy arrays
        if len(self.train_losses) == 0:
            self.val_accs, self.val_losses = [], []

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
            for idx, batch in enumerate(self.val_loader):
                if idx % print_every == 0:
                    self.logger.info(
                        f"Validation step: {idx}/{len(self.val_loader)} | {(idx/len(self.val_loader) * 100):.2f} %"
                    )

                # Check for partial maps
                keypoints = None
                if len(batch) == 3:
                    inputs, targets, keypoints = batch
                    self.validation_part_annotations.append(keypoints)
                else:
                    inputs, targets = batch
                
                self.validation_target_labels.append(targets)

                # Put the target to the target device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Call the evaluate step
                inference_time, step_loss, step_correct = self.evaluate_step(inputs, targets, keypoints=keypoints)

                # Update the train metrics
                running_loss += step_loss
                correct += step_correct
                total += targets.size(0)

                # Update the current best_time
                if inference_time < self.best_inference_time:
                    self.best_inference_time = inference_time

        # Calculate the evaluation stats
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct / total
        validation_msg = f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}"

        # Save the validation loss and accuracy
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

        # Calculate the consistency if possible
        if len(self.validation_activation_maps) > 0:
            # concatenate maps
            activation_maps = torch.cat(self.validation_activation_maps, dim=0).to(device=self.device)
            part_annotations = torch.cat(self.validation_part_annotations, dim=0).to(device=self.device)
            labels = torch.cat(self.validation_target_labels, dim=0).to(device=self.device)
            consistency = self.compute_consistency_score(
                activation_maps.to(self.device),
                part_annotations,
                labels,
                self.model.get_prototype_to_category()
            )
            validation_msg += f' | Validation consistency: {consistency:.4f}'
            self.val_consistencies.append(consistency)

        self.logger.info(validation_msg)

        return val_loss, val_acc
    
    def evaluate_step(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> tuple[float, float, int]:
        # Calculate the loss
        start_time = time.time()
        outputs = self.model(inputs)
        end_time = time.time()
        loss = self.criterion(outputs, targets)

        # Calculate the batch stats
        running_loss = loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()

        return end_time - start_time, running_loss, correct

    def _save_plots(self):
        """Function to save the plots from training."""
        self._save_loss_plot()
        self._save_acc_plot()
        self._save_consistency_plot()

    def _save_consistency_plot(self):
        """Generates and saves the plot for validation consistency."""
        if len(self.val_consistencies) == 0:
            pass
        
        # Get the plot path 
        save_path = self.output_path + "consistency.png"

        # Create the plot
        epochs = range(1, len(self.val_consistencies) + 1)
        plt.figure()
        plt.plot(epochs, self.val_consistencies, label='Validation Consistencies')
        plt.xlabel('Epoch')
        plt.ylabel('Consistency')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f'Consistency plot saved to {save_path}')

    def _save_loss_plot(self):
        """Generates and saves the plot for training and validation losses."""
        # Get the plot path 
        save_path = self.output_path + "loss_plot.png"

        # Create the plot
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
        self.logger.info(f'Loss plot saved to {save_path}')

    def _save_acc_plot(self):
        """Generates and saves the plot for training and validation accuracies."""
        # Get the plot path 
        save_path = self.output_path + "acc_plot.png"

        # Create the plot
        epochs = range(1, len(self.train_accs) + 1)
        plt.figure()
        plt.plot(epochs, self.train_accs, label='Train Acc')
        if self.val_loader:
            plt.plot(epochs, self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f'Accuracy plot saved to {save_path}')

    def _save_inference_stats(self):
        """Saves model parameter count and best inference time to a txt file."""
        # Get the inderence path
        save_path = self.output_path + "inference_stats.txt"

        # Save the stats
        with open(save_path, 'w') as f:
            f.write(f"Total Parameters: {self.total_param_elements}\n")
            if self.best_inference_time is not None:
                f.write(f"Best Inference Time (s): {self.best_inference_time:.6f}\n")
            else:
                f.write("Inference time not measured.\n")
        self.logger.info(f'Inference stats saved to {save_path}')

    def _save_checkpoint(self):
        """Saves model and optimizer state to a checkpoint file."""
        # Get save_path
        save_path = self.output_path + "model_checkpoint.pth"

        # Save the state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        self.logger.info(f'Checkpoint saved to {save_path}')

    def compute_consistency_score(
        self,
        act_maps: torch.Tensor,
        keypoints: torch.Tensor,
        batch_labels: torch.Tensor,
        prototype_to_category: torch.Tensor,
    ) -> float:
        """
        Function to compute the consistency.
        
        Args:
            act_maps:    [B, M, H, W] distance maps from prototype_layer
            keypoints:   [B, C, 4] with (i_part, x_coord, y_coord, vis_flag)
            batch_labels: [B] category labels for each image
            prototype_to_category: [M] allocated category index for each prototype
        Returns:
            S_con ∈ [0,1]
        """
        _, M, _, _ = act_maps.shape
        H, W = self.image_size
        H_b, W_b = self.box_size
        H2, W2 = H_b // 2, W_b // 2
        C = keypoints.shape[1]  # Number of part categories
        
        # Precompute keypoint tensors
        x_map = keypoints[..., 1]
        y_map = keypoints[..., 2]
        vis_map = keypoints[..., 3]
        
        # Initialize accumulation tensors
        sums = torch.zeros(M, C, device=act_maps.device)
        counts = torch.zeros(M, device=act_maps.device)
        
        # Process each prototype sequentially
        for j in range(M):
            # Get current prototype's activation maps
            act_j = act_maps[:, j]  # [B, H_act, W_act]
            
            # Find images belonging to this prototype's category
            cat_mask = (batch_labels == prototype_to_category[j])
            if not cat_mask.any():
                continue
                
            # Filter relevant images
            act_j = act_j[cat_mask]
            x_map_j = x_map[cat_mask]
            y_map_j = y_map[cat_mask]
            vis_map_j = vis_map[cat_mask]
            B_j = act_j.shape[0]  # Reduced batch size
            
            # Upsample only for this prototype
            act_j_up = torch.nn.functional.interpolate(
                act_j.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B_j, H, W]
            
            # Find winning locations
            flat_j = act_j_up.view(B_j, -1)
            _, min_idx = flat_j.min(dim=1)
            y_c = min_idx // W
            x_c = min_idx % W
            
            # Compute boxes [B_j, 1]
            y1 = (y_c - H2).clamp(0, H - H_b).unsqueeze(1)
            x1 = (x_c - W2).clamp(0, W - W_b).unsqueeze(1)
            y2 = y1 + H_b
            x2 = x1 + W_b
            
            # Check keypoint containment [B_j, C]
            in_x = (x_map_j >= x1) & (x_map_j < x2)
            in_y = (y_map_j >= y1) & (y_map_j < y2)
            op = (in_x & in_y & vis_map_j.bool()).float()
            
            # Update accumulators
            sums[j] = op.sum(dim=0)
            counts[j] = B_j
        
        # Compute consistency scores per prototype
        a_j = sums / counts.clamp(min=1).unsqueeze(-1)  # [M, C]
        max_scores = a_j.max(dim=1).values  # [M]
        consistent = (max_scores >= self.consistency_treshold).float()
        
        return consistent.mean().item()
