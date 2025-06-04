import argparse
import torch
import numpy as np
from logging import Logger
from torch.utils.data import DataLoader

from tasks.prototree.model import ProtoTree

def project_with_class_constraints(
        tree: ProtoTree,
        project_loader: DataLoader,
        device: str,
        logger: Logger,  
    ) -> tuple[dict, ProtoTree]:
    logger.info("Projecting prototypes to nearest training patch (with class restrictions)...")
    
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()

    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        
        # For each internal node, collect the leaf labels in the subtree with this node as root. 
        leaf_labels_subtree = dict()
        for branch, j in tree._out_map.items():
            leaf_labels_subtree[branch.index()] = set()
            for leaf in branch.leaves():
                leaf_labels_subtree[branch.index()].add(torch.argmax(leaf.distribution()).item())
        
        for i, (xs, ys) in enumerate(project_loader):
            xs, ys = xs.to(device), ys.to(device)

            # Get the features and distances
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            _, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():
                leaf_labels = leaf_labels_subtree[node.index()]

                # Select the features/distances that are relevant to this prototype
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                    #Check if label of this image is in one of the leaves of the subtree
                    if ys[batch_i].item() in leaf_labels: 
                        # Find the index of the latent patch that is closest to the prototype
                        min_distance = distances.min()
                        min_distance_ix = distances.argmin()

                        # Use the index to get the closest latent patch
                        closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                        # Check if the latent patch is closest for all data samples seen so far
                        if min_distance < global_min_proto_dist[j]:
                            global_min_proto_dist[j] = min_distance
                            global_min_patches[j] = closest_patch
                            global_min_info[j] = {
                                'input_image_ix': i * batch_size + batch_i,
                                'patch_ix': min_distance_ix.item(),
                                'W': W,
                                'H': H,
                                'W1': W1,
                                'H1': H1,
                                'distance': min_distance.item(),
                                'nearest_input': torch.unsqueeze(xs[batch_i],0),
                                'node_ix': node.index(),
                            }

        # Copy the patches to the prototype layer weights
        torch.cat(
            tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
            dim=0,
            out=tree.prototype_layer.prototype_vectors
        )

    return global_min_info, tree