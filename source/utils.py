import numpy as np
import torch
from torch.nn import functional as F

def vq(input, codebook):
    '''
    input: (b, d)
    codeebook: (K, c)
    return: (b,)
    '''
    input = input.unsqueeze(1)  # (b, 1, d)
    codebook = codebook.unsqueeze(0) # (1, l, d)
    distances = ((input - codebook) ** 2).sum(-1)   # (b, l, d) --> (b, l)
    _, indices = distances.min(-1)
    return indices

def denormalize(img):
    img = img * 0.5 + 0.5
    return img
