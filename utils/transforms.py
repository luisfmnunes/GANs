import torch
from torch.autograd import Variable

def images_to_vectors(images : torch.Tensor) -> torch.Tensor:
    """
    Flattens an image array into 1-D Tensor representations
    """
    return images.view(images.size(0), 784) # 28x28 -> 1x784

def vectors_to_images(vectors : torch.Tensor) -> torch.Tensor:
    """
    Transform flatten tensors (1-D) into 28x28 images (NCHW)
    """
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    """
    Generates a 1-D vector of gaussian sampled random values
    """
    n = Variable(torch.randn(size,100))
    return n
