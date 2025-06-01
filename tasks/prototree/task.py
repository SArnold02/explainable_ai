from argparse import Namespace
import torch
import torchvision
from torchvision.models import ResNet34_Weights

from tasks.baseline.task import ResNet
from tasks.prototree.model import ProtoTree
from .tree_trainer import ProtoTreeTrainer
from .custom_cub_dataset import Cub2011, DEFAULT_VAL_TRANSFORM, DEFAULT_TRAIN_TRANSFORM

class CustomResNet(torch.nn.Module):
    """Custom ResNet wrapper to take out the flatten function call."""
    def __init__(self, model: ResNet):
        super().__init__()
        self.model = model.base_model

    def forward(self, model_input: torch.Tensor):
        x = self.model.conv1(model_input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

def get_pre_trained_prototree(arguments) -> torch.nn.Module:
    # Build the backbone
    weights = ResNet34_Weights if arguments.resnet_checkpoint is None and arguments.checkpoint is None else None
    model = ResNet(weights=weights, num_classes=arguments.num_classes)
    
    # Check if the resnet model checkpoint is given
    if arguments.resnet_checkpoint is not None:
        checkpoint = torch.load(arguments.resnet_checkpoint, weights_only=False, map_location=arguments.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Create the resnet with without the final layers
    model = CustomResNet(model)

    # Build the ProtTree
    tree = ProtoTree(
        model,
        arguments
    )

    # Load the checkpoint 
    if arguments.checkpoint is not None:
        checkpoint = torch.load(arguments.checkpoint, weights_only=False, map_location=arguments.device)
        tree.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Put the model to device
    tree.to(arguments.device)
    return tree

def get_optimizer(tree: ProtoTree, args: Namespace) -> tuple[torch.optim.Optimizer, list[torch.Tensor]]:
    """Construct the optimizer as dictated by the parsed arguments."""
    # Initialize the parameter groups
    params_to_freeze = []
    dist_params = []

    # Get the dist params
    for name,param in tree.named_parameters():
        if 'dist_params' in name:
            dist_params.append(param)
    
    # Freeze the backbone parameters
    for name, param in tree._net.named_parameters():
        params_to_freeze.append(param)

    # Create the AdamW optimizer
    paramlist = [
        {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay}, 
        {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
        {"params": tree.prototype_layer.parameters(), "lr": args.lr,"weight_decay_rate": 0}]
    if not args.kont_algorithm:
        paramlist.append({"params": dist_params, "lr": args.lr, "weight_decay_rate": 0})
    
    return torch.optim.AdamW(paramlist, lr=args.lr, eps=1e-07, weight_decay=args.weight_decay), params_to_freeze

def run_prototree_train(arguments, train_dataset, val_dataset):
    # Replace the datasets
    if arguments.dataset == "cub":
        train_dataset = Cub2011(train=True)
        val_dataset = Cub2011(train=False)
    else:
        train_dataset = torchvision.datasets.ImageFolder("data/stanford_carstrain", transform=DEFAULT_TRAIN_TRANSFORM)
        val_dataset = torchvision.datasets.ImageFolder("data/stanford_carstest", transform=DEFAULT_VAL_TRANSFORM)
    
    # Build the pre-trained ResNet
    model = get_pre_trained_prototree(arguments)

    # Build the loss and the optimizer for the tree
    criterion = torch.nn.NLLLoss()
    optimizer, params_to_freeze = get_optimizer(model, arguments)

    # Create the trainer
    trainer = ProtoTreeTrainer(
        model = model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        arguments = arguments,
        criterion = criterion,
        optimizer = optimizer,
        params_to_freeze = params_to_freeze,
    )

    # Start the training
    trainer.train(
        epochs = arguments.epochs,
        print_every = arguments.print_every,
    )

    # Prune the tree
    trainer.prune(0.01)

def run_prototree_eval(arguments, val_dataset):
    # Replace the datasets
    if arguments.dataset == "cub":
        val_dataset = Cub2011(train=False)
    else:
        val_dataset = torchvision.datasets.ImageFolder("data/stanford_carstest", transform=DEFAULT_VAL_TRANSFORM)

    # Build the pre-trained ResNet
    model = get_pre_trained_prototree(arguments)

    # Create the trainer
    trainer = ProtoTreeTrainer(
        model = model,
        train_dataset = None,
        val_dataset = val_dataset,
        arguments = arguments,
    )

    # Start the evaluation
    trainer.evaluate(
        print_every = arguments.print_every,
    )

def run_prototree(arguments, train_dataset, val_dataset):
    # Choose the type of run
    if arguments.train_run:
        run_prototree_train(arguments, train_dataset, val_dataset)
    else:
        run_prototree_eval(arguments, val_dataset)