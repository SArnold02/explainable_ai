import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from common import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms

from .tesnet_model import TesNet
from ..baseline.task import get_pre_trained_baseline, ResNet

matplotlib.use('Agg')


def run_tesnet(arguments, train_dataset, val_dataset):
    """
    Runs training or testing of the model based on parameters.
    :param arguments: The arguments.
    :param train_dataset: The training set.
    :param val_dataset: The validation set.
    """
    if arguments.train_run:
        run_train(arguments, train_dataset, val_dataset)
    else:
        run_eval(arguments, val_dataset)


def run_train(arguments, train_dataset, val_dataset):
    """
    Trains the model.
    :param arguments: The arguments.
    :param train_dataset: The training set.
    :param val_dataset: The validation set.
    """

    backbone = get_pre_trained_baseline(arguments)
    model = TesNet(
        backbone=backbone,
        num_classes=arguments.num_classes
    )

    device = arguments.device

    # Custom loss wrapper
    class TesNetLoss(nn.Module):
        def __init__(self, alpha=1.0, beta=0.1):
            super().__init__()
            self.ce_loss = nn.CrossEntropyLoss()

        def compute_ortho_loss(self):
            """
            Push within-class vectors apart using Frobenius norm.

            :return: The orthonormality loss.
            """
            prototypes = model.prototype_vectors
            P, D = prototypes.size(0), prototypes.size(1)
            prototypes = prototypes.view(model.num_classes, model.num_prototypes_per_class, D)
            product = torch.matmul(prototypes, prototypes.transpose(1, 2))
            identity = torch.eye(model.num_prototypes_per_class, device=device).unsqueeze(0)
            diff = product - identity
            orth_loss = torch.sum(torch.relu(torch.norm(diff, p=1, dim=[1, 2]) - 0))
            return orth_loss

        def compute_separation_loss(self):
            """
            Separate subspaces of different classes using Projection Metric.

            :return: The separation loss.
            """
            prototypes = model.prototype_vectors
            P, D = prototypes.size(0), prototypes.size(1)
            prototypes = prototypes.view(model.num_classes, model.num_prototypes_per_class, D)
            projection_operator = torch.matmul(prototypes, prototypes.transpose(1, 2))

            projection_operator_1 = torch.unsqueeze(projection_operator, dim=1)
            projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)
            pairwise_distance = torch.norm(projection_operator_1 - projection_operator_2 + 1e-10, p='fro', dim=[2, 3])
            subspace_sep = 0.5 * torch.norm(pairwise_distance, p=1, dim=[0, 1], dtype=torch.double) / torch.sqrt(
                torch.tensor(2, dtype=torch.double)).to(device)
            return subspace_sep

        def compute_grouping_loss(self, labels):
            """
            Encourage patches to be close to at least one semantically similar basis vector of the ground truth.

            :param labels: The ground truth.
            :return: The clustering loss.
            """
            correct_prototypes = torch.t(model.prototype_class_identity[:, labels]).to(device)
            inverted_dist, _ = torch.max((model.prototype_shape[1] - model.cosine_min_distances) * correct_prototypes,
                                         dim=1)
            cluster_loss = torch.mean(model.prototype_shape[1] - inverted_dist)

            return cluster_loss

        def forward(self, outputs, labels):
            """
            Compute the loss.
            :param outputs: Classification outputs.
            :param labels: Ground truth labels.
            :return: The final loss.
            """
            # Classification loss
            ce_loss = self.ce_loss(outputs, labels)
            ortho_loss = self.compute_ortho_loss()
            sep_loss = self.compute_separation_loss()
            cluster_loss = self.compute_grouping_loss(labels)

            final_loss = ce_loss + 0.8 * cluster_loss + 1e-4 * ortho_loss - 1e-7 * sep_loss

            return final_loss

    # Trainer with TesNet and custom loss
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        arguments=arguments,
        criterion=TesNetLoss()
    )

    # Start training
    trainer.train(
        epochs=arguments.epochs,
        print_every=arguments.print_every,
    )


def run_eval(arguments, val_dataset):
    """
    Tests the model.
    :param arguments: The arguments.
    :param val_dataset: The validation dataset.
    """

    model = TesNet(ResNet(None, arguments.num_classes), arguments.num_classes)
    checkpoint = torch.load(arguments.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = arguments.device

    # trainer = Trainer(
    #     model=model,
    #     train_dataset=None,
    #     val_dataset=val_dataset,
    #     arguments=arguments,
    # )
    #
    # # Start the evaluation
    # trainer.evaluate(
    #     print_every=arguments.print_every,
    # )

    custom_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_dataset.transform = custom_transform
    visualize_top_activated_prototype(model, val_dataset, device, "cars_dataset.png")


def visualize_top_activated_prototype(model, dataset, device, save_path, nr_imgs=8, layer_name='conv_1x1'):
    """
    Visualize the most activated prototype's heatmap over an input image.
    :param model: The model.
    :param dataset: The dataset.
    :param device: The device.
    :param layer_name: The layer to extract activation from.
    :return: The visualization.
    """

    model.eval()
    activations = {}

    def hook_fn(module, input, output):
        activations["features"] = output.detach()

    # Register the hook
    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook_fn)

    num_images = min(nr_imgs, len(dataset))
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    for idx in range(num_images):
        image_tensor, _ = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(input_tensor)

        fmap = activations["features"]
        cosine_dists = model.cosine_dist(fmap)  # [1, num_prototypes, Hf, Wf]

        proto_map = cosine_dists[0]  # [num_prototypes, Hf, Wf]
        min_dist = proto_map.view(proto_map.size(0), -1).min(dim=1).values
        top_proto = torch.argmin(min_dist).item()

        selected_map = proto_map[top_proto]
        selected_map = (selected_map - selected_map.min()) / (selected_map.max() - selected_map.min() + 1e-6)

        H, W = image_tensor.shape[1:]  # [C, H, W]
        heatmap = cv2.resize(selected_map.detach().numpy(), (W, H))

        img_np = TF.to_pil_image(image_tensor).convert("RGB")
        img_np = np.array(img_np)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx]
        ax.imshow(overlayed)
        ax.set_title(f"Img {idx}, Proto {top_proto}")
        ax.axis("off")

    # Hide any unused subplots
    for j in range(num_images, rows * cols):
        ax = axes[j // cols, j % cols] if rows > 1 else axes[j]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    handle.remove()
    Image.open(save_path).show()


def extract_topk_patches(model, dataset, proto_idx, device, k=5, patch_size=64):
    """
    Get top-K image patches for a prototype.
    :param model: The model.
    :param dataset: The dataset.
    :param proto_idx: The prototype index.
    :param k: Number of patches.
    :param patch_size: Patch size.
    :return: The visualization of the k patches.
    """
    dataset = torch.utils.data.Subset(dataset, range(10))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    top_activations = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            conv_act = activation["conv_1x1"]
            cosine_dists = model.cosine_dist(conv_act)

            proto_act = cosine_dists[:, proto_idx]
            values, indices = torch.max(proto_act.view(proto_act.size(0), -1), dim=1)

            for i in range(images.size(0)):
                top_activations.append({
                    "value": values[i].item(),
                    "image": images[i].cpu(),
                    "feature_map": proto_act[i].cpu(),
                    "original_idx": batch_idx * loader.batch_size + i,
                    "location": np.unravel_index(indices[i].item(), proto_act[i].shape)
                })

    # Sort by strongest activation (lowest distance)
    top_activations = sorted(top_activations, key=lambda x: -x["value"])[:k]

    # Crop and display patches
    fig, axs = plt.subplots(1, k, figsize=(15, 3))
    for i, entry in enumerate(top_activations):
        image = TF.to_pil_image(entry["image"])
        H, W = image.size
        fmap_h, fmap_w = entry["feature_map"].shape
        h_ratio = H / fmap_h
        w_ratio = W / fmap_w
        y, x = entry["location"]

        # Convert feature map location to pixel coords
        cx, cy = int((x + 0.5) * w_ratio), int((y + 0.5) * h_ratio)
        half = patch_size // 2
        left = max(cx - half, 0)
        upper = max(cy - half, 0)
        right = min(cx + half, W)
        lower = min(cy + half, H)

        crop = image.crop((left, upper, right, lower))
        axs[i].imshow(crop)
        axs[i].set_title(f"#{entry['original_idx']}")
        axs[i].axis("off")

    plt.suptitle(f"Top-{k} patches for Prototype {proto_idx}")
    plt.tight_layout()
    plt.savefig("activation_map.png")

    Image.open("activation_map.png").show()
