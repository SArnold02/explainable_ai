import torch
from torchvision.models import resnet34, ResNet34_Weights
from common import Trainer

class ResNet(torch.nn.Module):
    def __init__(self, weights, num_classes = 200):
        # Change the final classification head
        super().__init__()
        self.base_model = resnet34(weights=weights)
        self.base_model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = torch.nn.Linear(512, num_classes)

        self._weight_init()

    def forward(self, model_inputs: torch.Tensor):
        # Call the forward for the changed model
        return self.base_model(model_inputs)

    def _weight_init(self):
        torch.nn.init.kaiming_normal_(self.base_model.fc.weight.data)

def get_pre_trained_baseline(arguments) -> torch.nn.Module:
    # Build the model
    model = ResNet(weights=None, num_classes=arguments.num_classes)

    # Load the checkpoint 
    assert arguments.checkpoint is not None, "No checkpoint given"
    checkpoint = torch.load(arguments.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def run_baseline_train(arguments, train_dataset, val_dataset):
    # Build ResNet-34 model
    if arguments.checkpoint is not None:
        model = get_pre_trained_baseline(arguments)
    else:
        model = ResNet(weights=ResNet34_Weights.DEFAULT, num_classes=arguments.num_classes)

    # Create the trainer
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        arguments = arguments,
    )

    # Start the training
    trainer.train(
        epochs = arguments.epochs,
        print_every = arguments.print_every,
    )

def run_baseline_eval(arguments, val_dataset):
    # Build the pre-trained ResNet
    model = get_pre_trained_baseline(arguments)

    # Create the trainer
    trainer = Trainer(
        model = model,
        train_dataset = None,
        val_dataset = val_dataset,
        arguments = arguments,
    )

    # Start the evaluation
    trainer.evaluate(
        print_every = arguments.print_every,
    )

def run_baseline(arguments, train_dataset, val_dataset):
    # Choose the type of run
    if arguments.train_run:
        run_baseline_train(arguments, train_dataset, val_dataset)
    else:
        run_baseline_eval(arguments, val_dataset)