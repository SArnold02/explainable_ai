
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeInterface(nn.Module):
    # Present for python helper
    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError
    
    def leaves(self) -> set:
        raise NotImplementedError

    def branches(self) -> set:
        raise NotImplementedError
    
    def nodes_by_index(self) -> dict:
        raise NotImplementedError
    
    def num_branches(self) -> int:
        raise NotImplementedError
    
    def num_leaves(self) -> int:
        raise NotImplementedError
    
    def nodes(self) -> set:
        raise NotImplementedError
    
class Node(NodeInterface):
    def __init__(self,
                 index: int,
                 arguments: Namespace,
                 is_leaf: bool = False,
                 left_node: NodeInterface | None = None,
                 right_node: NodeInterface | None = None,
                 ) -> None:
        # Initialize the torch module
        super().__init__()
        self._index = index
        self.is_leaf_node = is_leaf
        self.kont_algorithm = arguments.kont_algorithm
        self._log_probabilities = arguments.log_probabilities

        self.left_node = left_node
        self.right_node = right_node

        # Initialize the distribution parameter
        if self.kont_algorithm:
            self._dist_params = nn.Parameter(torch.ones(arguments.num_classes), requires_grad=False)
        else:
            self._dist_params = nn.Parameter(torch.zeros(arguments.num_classes), requires_grad=True)

    def forward(self, node_input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        # Decide which forward implementation to call
        if self.is_leaf_node:
            return self._forward_leaf_implementation(node_input, **kwargs)
        else:
            return self._forward_normal_implementation(node_input, **kwargs)

    def _forward_leaf_implementation(self, node_input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        # Get the batch size
        batch_size = node_input.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        # In this dict, store the probability of arriving at this node.
        # If not set, set the probability of arriving at this node as one
        node_attr = kwargs.setdefault('attr', dict())
        if not self._log_probabilities:
            node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=node_input.device))
        else:
            node_attr.setdefault((self, 'pa'), torch.zeros(batch_size, device=node_input.device))

        # Calculate the distribution
        dist = self.distribution()
        dist = dist.view(1, -1)
        expanded_dist = dist.repeat((batch_size, 1))

        # Store the leaf distribution
        node_attr[self, 'ds'] = expanded_dist

        return expanded_dist, node_attr

    def _forward_normal_implementation(self, node_input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        # Get the batch size
        batch_size = node_input.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        # In this dict, store the probability of arriving at this node.
        # If not set, set the probability of arriving at this node as one
        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=node_input.device))

        # Calculate the probability of taking the right subtree
        ps = self.get_tree_probability(**kwargs)

        # Store decision node probabilities as node attribute
        node_attr[self, 'ps'] = ps
        
        if not self._log_probabilities:
            # Store path probabilities of arriving at child nodes as node attributes
            node_attr[self.left_node, 'pa'] = (1 - ps) * pa
            node_attr[self.right_node, 'pa'] = ps * pa

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.left_node.forward(node_input, **kwargs)
            r_dists, _ = self.right_node.forward(node_input, **kwargs)

            # Weight the probability distributions by the decision node's output
            ps = ps.view(batch_size, 1)
            return (1 - ps) * l_dists + ps * r_dists, node_attr

        # For log probability calculations
        x = torch.abs(ps) + 1e-7  # add small epsilon for numerical stability
        oneminusp = torch.where(x < torch.log(torch.tensor(2)), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

        node_attr[self.left_node, 'pa'] = oneminusp + pa
        node_attr[self.right_node, 'pa'] = ps + pa

        # Obtain the unweighted probability distributions from the child nodes
        l_dists, _ = self.left_node.forward(node_input, **kwargs)
        r_dists, _ = self.right_node.forward(node_input, **kwargs)

        # Weight the probability distributions by the decision node's output
        ps = ps.view(batch_size, 1)
        oneminusp = oneminusp.view(batch_size, 1)
        logs_stacked = torch.stack((oneminusp + l_dists, ps + r_dists))
        return torch.logsumexp(logs_stacked, dim=0), node_attr 

    def get_tree_probability(self, **kwargs) -> torch.Tensor:
        # Obtain the output corresponding to this decision node
        out_map = kwargs['out_map']
        conv_net_output = kwargs['conv_net_output']
        out = conv_net_output[out_map[self]]
        return out.squeeze(dim=1)
    
    def distribution(self) -> torch.Tensor:
        # Choose the right distribution calculation for the run
        if not self.kont_algorithm:
            if self._log_probabilities:
                return F.log_softmax(self._dist_params, dim=0)
            else:
                return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)
        else:
            if self._log_probabilities:
                return torch.log((self._dist_params / torch.sum(self._dist_params))+1e-10)
            else:
                return (self._dist_params / torch.sum(self._dist_params))

    def index(self) -> int:
        return self._index
    
    def size(self) -> int:
        size = 1
        if self.left_node:
            size += self.left_node.size()
        if self.right_node:
            size += self.right_node.size()
        return size

    def depth(self) -> int:
        depth = 1
        left_depth, rigth_depth = 0, 0
        if self.left_node:
            left_depth = self.left_node.depth()
        if self.right_node:
            rigth_depth = self.right_node.depth()
        return max(depth, left_depth, rigth_depth)
    
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    def leaves(self) -> set:
        leaves = {self} if self.is_leaf_node else set()
        if self.left_node:
            leaves = leaves.union(self.left_node.leaves())
        if self.right_node:
            leaves = leaves.union(self.right_node.leaves())

        return leaves
    
    def branches(self) -> set:
        branches = {self} if not self.is_leaf_node else set()
        if self.left_node:
            branches = branches.union(self.left_node.branches())
        if self.right_node:
            branches = branches.union(self.right_node.branches())

        return branches
    
    def nodes_by_index(self) -> dict:
        nodes_by_index = {self.index(): self}
        if self.left_node:
            nodes_by_index.update(self.left_node.nodes_by_index())
        if self.right_node:
            nodes_by_index.update(self.right_node.nodes_by_index())

        return nodes_by_index
    
    def num_branches(self) -> int:
        num_branches = 1 if not self.is_leaf_node else 0
        if self.left_node:
            num_branches += self.left_node.num_branches()
        if self.right_node:
            num_branches += self.right_node.num_branches()

        return num_branches
    
    def num_leaves(self) -> int:
        num_leaves = 1 if self.is_leaf_node else 0
        if self.left_node:
            num_leaves += self.left_node.num_leaves()
        if self.right_node:
            num_leaves += self.right_node.num_leaves()

        return num_leaves
    
    def nodes(self) -> set:
        return self.branches().union(self.leaves())