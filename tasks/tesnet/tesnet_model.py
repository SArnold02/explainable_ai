import torch
import torch.nn as nn
import torch.nn.functional as F


class TesNet(torch.nn.Module):
    def __init__(self, backbone, num_classes=200, prototypes_per_class=10, prototype_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.prototype_shape = (num_classes * prototypes_per_class, prototype_dim, 1, 1)
        self.num_prototypes = self.prototype_shape[0]

        assert (self.num_prototypes % self.num_classes == 0)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        # Backbone
        self.backbone = backbone

        # add 1x1 conv
        for module in reversed(list(self.backbone.modules())):
            if isinstance(module, nn.Conv2d):
                in_channels = module.out_channels
                break


        out_channels = self.prototype_shape[1]
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv_1x1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv_1x1.bias is not None:
            nn.init.constant_(self.conv_1x1.bias, 0)

        # Prototype vectors
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.prototype_class_identity = self._create_class_identity()

        # Final linear layer mapping prototype scores to class logits
        self.classifier = torch.nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        self.cosine_min_distances = None

    def _create_class_identity(self):
        identity = torch.zeros(self.num_prototypes, self.num_classes)
        for i in range(self.num_prototypes):
            identity[i, i // self.num_prototypes_per_class] = 1
        return identity

    def cosine_dist(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        distances = F.conv2d(input=x_norm, weight=self.prototype_vectors)

        return -distances

    def project_to_basis(self, x):
        distances = F.conv2d(input=x, weight=self.prototype_vectors)
        return distances

    def apply_resnet_conv(self, x):
        conv_output = self.backbone.base_model.conv1(x)
        conv_output = self.backbone.base_model.bn1(conv_output)
        conv_output = self.backbone.base_model.relu(conv_output)
        conv_output = self.backbone.base_model.maxpool(conv_output)

        conv_output = self.backbone.base_model.layer1(conv_output)
        conv_output = self.backbone.base_model.layer2(conv_output)
        conv_output = self.backbone.base_model.layer3(conv_output)
        conv_output = self.backbone.base_model.layer4(conv_output)

        return conv_output

    def forward(self, x):

        self.prototype_vectors.data = F.normalize(self.prototype_vectors, p=2, dim=1).data

        conv_output = self.apply_resnet_conv(x)
        conv_output = self.conv_1x1(conv_output)

        # similarity between each feature vector and each prototype
        cosine_dist = self.cosine_dist(conv_output)
        # find most similar patch per prototype (smallest distance)
        self.cosine_min_distances = F.max_pool2d(cosine_dist,
                                                  kernel_size=(cosine_dist.size()[2], cosine_dist.size()[3]))
        self.cosine_min_distances = self.cosine_min_distances.view(-1, self.num_prototypes)

        # project feature vectors onto basis vectors (orthogonal projection)
        project_dist = self.project_to_basis(conv_output)
        # find strongest evidence for a concept
        project_max_dist = F.max_pool2d(project_dist, kernel_size=(project_dist.size()[2], project_dist.size()[3]))
        project_max_dist = project_max_dist.view(-1, self.num_prototypes)

        logits = self.classifier(project_max_dist)

        return logits
