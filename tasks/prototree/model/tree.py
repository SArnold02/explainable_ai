import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.prototree.model.node import Node
from tasks.prototree.model.l2conv import L2Conv2D

class ProtoTree(nn.Module):
    def __init__(self,
                 feature_net: torch.nn.Module,
                 arguments: argparse.Namespace,
                 ) -> None:
        # Initialize the ProtoTree
        super().__init__()
        self._num_classes = arguments.num_classes
        self._log_probabilities = arguments.log_probabilities

        # Build the tree
        self._root = self._init_tree(arguments)

        self.num_features = arguments.num_features
        self.num_prototypes = self.num_branches()
        self.prototype_shape = (arguments.W1, arguments.H1, arguments.num_features)
        
        # Keep a dict that stores a reference to each node's parent
        self._parents = dict()
        self._set_parents()

        # Set the feature network
        self._net = feature_net
        in_channels = [i for i in self._net.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        self._add_on = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.num_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        ) 

        # Map each decision node to an output of the feature net
        self._out_map = {n: i for i, n in zip(range(2 ** (arguments.depth) - 1), self.branches())}

        # Create the prototype layer
        self.prototype_layer = L2Conv2D(
            self.num_prototypes,
            self.num_features,
            arguments.W1,
            arguments.H1
        )

        # Initialize the network parameters
        self._init_network()

    def _init_network(self) -> None:
        with torch.no_grad():
            # Initialize the newtork elements
            torch.nn.init.normal_(self.prototype_layer.prototype_vectors, mean=0.5, std=0.1)
            self._add_on.apply(init_weights_xavier)

    def forward(self, tree_input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        # Perform a forward pass with the conv net
        features = self._net(tree_input)
        features = self._add_on(features)
        batch_size, _, width, height = features.shape

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)

        # Perform global min pooling to see the minimal distance for each prototype to any patch of the input image
        min_distances = -F.max_pool2d(-distances, kernel_size=(width, height))
        min_distances = min_distances.view(batch_size, self.num_prototypes)
        if not self._log_probabilities:
            similarities = torch.exp(-min_distances)
        else:
            similarities = -min_distances

        # Add the conv net output to the kwargs dict to be passed to the decision nodes in the tree
        # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        kwargs['conv_net_output'] = similarities.chunk(similarities.size(1), dim=1)
        # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
        kwargs['out_map'] = dict(self._out_map)

        # Perform a forward pass through the tree
        out, attr = self._root(tree_input, **kwargs)

        # Store the probability of arriving at all nodes in the decision tree and the probabilities of decision nodes
        info = dict()
        info['pa_tensor'] = {n.index(): attr[n, 'pa'].unsqueeze(1) for n in self.nodes()}
        info['ps'] = {n.index(): attr[n, 'ps'].unsqueeze(1) for n in self.branches()}
        info['conv_net_output'] = distances

        return out, info
    
    def forward_partial(self, tree_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Perform a forward pass with the conv net
        features = self._net(tree_input)
        features = self._add_on(features)

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)

        return features, distances, dict(self._out_map)
    
    def root(self) -> Node:
        return self._root

    def depth(self) -> int:
        return self._root.depth()

    def size(self) -> int:
        return self._root.size()

    def nodes(self) -> set:
        return self._root.nodes()

    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index()

    def branches(self) -> set:
        return self._root.branches()

    def leaves(self) -> set:
        return self._root.leaves()

    def num_branches(self) -> int:
        return self._root.num_branches()

    def num_leaves(self) -> int:
        return self._root.num_leaves()

    def _init_tree(self, arguments: argparse.Namespace) -> Node:
        # Recursively build the tree
        def recursive_init(i: int, d: int) -> Node:
            if d == arguments.depth:
                return Node(
                    i,
                    arguments,
                    is_leaf=True
                )
            else:
                left = recursive_init(i + 1, d + 1)
                return Node(
                    i,
                    arguments,
                    is_leaf=False,
                    left_node=left,
                    right_node=recursive_init(i + left.size() + 1, d + 1),
                )

        return recursive_init(0, 0)

    def _set_parents(self) -> None:
        # Initialize the parent dictionary
        self._parents.clear()
        self._parents[self._root] = None

        def set_recursive(node: Node):
            if not node.is_leaf_node:
                self._parents[node.right_node] = node
                self._parents[node.left_node] = node
                set_recursive(node.right_node)
                set_recursive(node.left_node)
                return

        # Set all parents by traversing the tree starting from the root
        set_recursive(self._root)

    def path_to(self, node: Node):
        # Return the path to one of the nodes from root
        path = [node]

        while isinstance(self._parents[node], Node):
            node = self._parents[node]
            path = [node] + path

        return path
    
    def get_leaves_in_subtree(self, node: Node) -> set:
        """Collect all leaf nodes in the subtree rooted at given node"""
        if node.is_leaf_node:
            return {node}
        leaves = set()
        if node.left_node:
            leaves = leaves.union(self.get_leaves_in_subtree(node.left_node))
        if node.right_node:
            leaves = leaves.union(self.get_leaves_in_subtree(node.right_node))
        return leaves
    
    def get_prototype_to_category(self) -> torch.Tensor:
        # Computes prototype-to-category mapping by aggregating leaf distributions in each decision node's subtree.
        prototype_to_category = torch.full(
            (self.num_prototypes,), 
            -1, 
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        
        # Iterate through all decision nodes
        for node in self.branches():
            # Get leaves under this decision node
            leaves = self.get_leaves_in_subtree(node)
            
            # Aggregate class distributions from leaves
            agg_dist = torch.zeros(self._num_classes, device=prototype_to_category.device)
            for leaf in leaves:
                agg_dist += leaf.distribution().detach()  # Use detached distribution
            
            # Assign most frequent class
            category = torch.argmax(agg_dist).item()
            
            # Get prototype index from output mapping
            proto_idx = self._out_map[node]
            prototype_to_category[proto_idx] = category
        
        return prototype_to_category

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))