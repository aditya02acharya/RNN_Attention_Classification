import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class Retina:
    """
    implementation of human like retina.

    Extracts a retina like glimpse `phi` around location `l`
    from an image `x`.

    It, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    x: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
    l: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
    g: size of the first square patch.
    k: number of patches to extract in the glimpse.
    s: scaling factor that controls the size of successive patches.
    """

    def __init__(self, g, k, s):
        self.logger = logging.getLogger(__name__)
        self.g = g
        self.k = k
        self.s = s

        self.logger.debug("Glimpse setting: size: %d, n patches: %d, scale: %d" % (int(g), int(k), int(s)))

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).

        :param x: a 4D Tensor of shape (B, H, W, C). The minibatch of images.
        :param l: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
        :return: a 4D Tensor of shape (B, k, g, g). The minibatch of images.
        """

        phi = []
        size = self.g

        # extract k patches of increasing size.
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        phi = phi.view(-1, self.k, self.g, self.g)
        # phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in `x`.

        :param x: a 4D Tensor of shape (B, C, H, W). The minibatch of images.
        :param l: a 2D Tensor of shape (B, 2). Contains normalized coordinates in the range [-1, 1].
        :param size: a scalar defining the size of the extracted patch.
        :return patch: a 4D Tensor of shape (B, C, size, size)
        """
        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, [int(size // 2), int(size // 2), int(size // 2), int(size // 2)], "constant", 0)

        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1]: end[i, 1], start[i, 0]: end[i, 0]])

        return torch.stack(patch)

    def denormalize(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()
