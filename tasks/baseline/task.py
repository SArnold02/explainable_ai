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

def run_baseline(arguments, train_dataset, val_dataset):
    # Build ResNet-34 model
    model = ResNet(weights=ResNet34_Weights.DEFAULT, num_classes=200)

    # Create the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        arguments = arguments,
    )

    # Start the training
    trainer.train(
        epochs=arguments.epochs,
        print_every=arguments.print_every,
    )