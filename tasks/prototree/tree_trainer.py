import time
import torch
from copy import deepcopy
from common import Trainer
from tasks.prototree.model import Node

def freeze(epoch: int, params_to_freeze: list, freeze_epoch: int, logger):
    # Function to freeze or unfreeze the parameters passed
    if freeze_epoch>0:
        # Freeze the backbone
        if epoch == 1:
            logger.info("Network frozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = False
        # Unfreeze the backbone
        elif epoch == freeze_epoch:
            logger.info("Network unfrozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = True

class ProtoTreeTrainer(Trainer):
    def __init__(self, model, arguments, train_dataset = None, val_dataset = None, optimizer = None, criterion = None, params_to_freeze = None) -> None:
        # Initialize the base trainer
        super().__init__(model, arguments, train_dataset, val_dataset, optimizer, criterion)

        # Save tree specific arguments
        self.epoch = -1
        self.kont_algorithm = arguments.kont_algorithm
        self.params_to_freeze = params_to_freeze
        self.freeze_epoch = arguments.freeze_epoch

        # Setup specific lr scheduler
        milestones = [i for i in range(35, arguments.epochs, 10)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=arguments.gamma)

    def train_step(self, inputs, targets, **kwargs):
        # Check if need to start the kont algorithm at the start of the epoch
        if kwargs.get("epoch", None) is not None and kwargs["epoch"] > self.epoch:
            self.epoch = kwargs["epoch"]
            freeze(self.epoch, self.params_to_freeze, self.freeze_epoch, self.logger)
            if self.kont_algorithm:
                with torch.no_grad():
                    self._old_dist_params = dict()
                    for leaf in self.model.leaves():
                        self._old_dist_params[leaf] = leaf._dist_params.detach().clone()
                    # Optimize class distributions in leafs
                    self.eye = torch.eye(self.model._num_classes).to(self.device)
                
        # Zero the gradients
        self.optimizer.zero_grad()

        # Calculate the loss
        outputs, info = self.model(inputs)
        if self.model._log_probabilities:
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(torch.log(outputs), targets)

        # Call the backward pass
        loss.backward()
        self.optimizer.step()

        if self.kont_algorithm:
            #Update leaves with derivate-free algorithm
            #Make sure the tree is in eval mode
            self.model.eval()
            with torch.no_grad():
                target = self.eye[targets]
                for leaf in self.model.leaves():  
                    if self.model._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index()] + leaf.distribution() + torch.log(target) - outputs, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index()] * leaf.distribution() * target)/outputs, dim=0)  
                    leaf._dist_params -= (self._old_dist_params[leaf]/len(self.train_loader))
                    # Make sure no negative values are present
                    torch.nn.functional.relu_(leaf._dist_params)
                    leaf._dist_params += update

        self.model.train()

        # Get the batch stats
        running_loss = loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()

        return running_loss, correct
    
    def evaluate_step(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> tuple[float, float, int]:
        # Calculate the loss
        start_time = time.time()
        outputs, info = self.model(inputs)
        end_time = time.time()
        if self.model._log_probabilities:
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(torch.log(outputs), targets)

        # Calculate the batch stats
        running_loss = loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()

        # Save the values for the consistency
        if kwargs.get("keypoints", None) is not None:
            self.validation_activation_maps.append(info["conv_net_output"])

        return end_time - start_time, running_loss, correct
    
    def has_max_prob_lower_threshold(self, node: Node, threshold: float):
        # Returns True when all the node's children have a max leaf value < threshold
        if not node.is_leaf_node:
            for leaf in node.leaves():
                if leaf._log_probabilities:
                    value = torch.max(torch.exp(leaf.distribution())).item()
                else:
                    value = torch.max(leaf.distribution()).item()
                if value > threshold:
                    return False
        else:
            if node._log_probabilities:
                value = torch.max(torch.exp(node.distribution())).item()
            else:
                value = torch.max(node.distribution()).item()
            if value > threshold:
                return False
        return True

    def get_nodes_to_prune(self, threshold: float) -> list:
        # Collect the nodes to prune, based on the treshold
        nodes_to_prune = []
        for node in self.model.nodes():
            if self.has_max_prob_lower_threshold(node, threshold):
                #prune everything below incl this node
                nodes_to_prune.append(node.index())
        return nodes_to_prune
    
    def prune(self, treshold: float) -> None:
        # Prune the tree at the end of training
        self.logger.info("Pruning...")
        self.logger.info("Before pruning: %s branches and %s leaves"%(self.model.num_branches (), self.model.num_leaves()))
        
        # Get number of prototypes and the nodes to prune
        num_prototypes_before = self.model.num_branches()
        node_idxs_to_prune = self.get_nodes_to_prune(treshold)
        to_prune = deepcopy(node_idxs_to_prune)

        # remove children from prune_list of nodes that would already be pruned
        for node_idx in node_idxs_to_prune:
            if not self.model.nodes_by_index()[node_idx].is_leaf_node:
                # Parent cannot be root since root would then be removed
                if node_idx > 0:
                    for child in self.model.nodes_by_index()[node_idx].nodes():
                        if child.index() in to_prune and child.index() != node_idx:
                            to_prune.remove(child.index())
        
        # Restructure the tree to take out the nodes which should be pruned
        for node_idx in to_prune:
            node = self.model.nodes_by_index()[node_idx]
            parent = self.model._parents[node]
            if parent is None:
                continue

            # Parent cannot be root since root would then be removed
            if parent.index() > 0:
                if node == parent.left_node:
                    if parent == self.model._parents[parent].left_node:
                        #make right child of parent the left child of parent of parent
                        self.model._parents[parent.right_node] = self.model._parents[parent]
                        self.model._parents[parent].left_node = parent.right_node
                    elif parent == self.model._parents[parent].right_node:
                        #make right child of parent the right child of parent of parent
                        self.model._parents[parent.right_node] = self.model._parents[parent]
                        self.model._parents[parent].right_node = parent.right_node

                elif node == parent.right_node:
                    if parent == self.model._parents[parent].left_node:
                        #make left child or parent the left child of parent of parent
                        self.model._parents[parent.left_node] = self.model._parents[parent]
                        self.model._parents[parent].left_node = parent.left_node
                    elif parent == self.model._parents[parent].right_node:
                        #make left child of parent the right child of parent of parent
                        self.model._parents[parent.left_node] = self.model._parents[parent]
                        self.model._parents[parent].right_node = parent.left_node

        self.logger.info("After pruning: %s branches and %s leaves"%(self.model.num_branches(), self.model.num_leaves()))
        self.logger.info("Fraction of prototypes pruned: %s"%((num_prototypes_before - self.model.num_branches()) / float(num_prototypes_before))+'\n')