
import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Conv2D(nn.Module):
    def __init__(self, num_prototypes: int, num_features: int, w_1: int, h_1: int) -> None:
        super().__init__()

        # Each prototype is a latent representation of shape (num_features, w_1, h_1)
        prototype_shape = (num_prototypes, num_features, w_1, h_1)
        self.prototype_vectors = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # Adapted from ProtoPNet
        # Computing ||xs - ps ||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
        # where ps is some prototype image

        # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
        # with weights set to 1 so each patch just has its values summed)
        ones = torch.ones_like(self.prototype_vectors, device=xs.device)
        xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)

        # Now compute ||ps||^2
        # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
        # squared L2 distance
        ps_squared_l2 = torch.sum(self.prototype_vectors ** 2,dim=(1, 2, 3))

        # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)

        # Compute xs * ps (for all patches in the input image)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)

        # Use the values to compute the squared L2 distance
        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        distance = torch.sqrt(torch.abs(distance) + 1e-14)
        
        if torch.isnan(distance).any():
            raise Exception('Nan value gotten with prototypes.')
        return distance