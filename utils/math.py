
import torch
import numpy as np


def tensor(a, dtype=None, device=None):
    if isinstance(a, torch.Tensor): return a.to(device, dtype)
    return torch.tensor(a, dtype=dtype, device=device, requires_grad=False)


def length(a, axis=0, keepdims=False):
    return (a**2).sum(axis, keepdims)**.5


def norm(a, axis=0):
    return a/length(a, axis, keepdims=True)


def rotmat(theta):
    m = torch.eye(4)
    m[0, 0] = np.cos(theta)
    m[0, 3] = -np.sin(theta)
    m[3, 0] = np.sin(theta)
    m[3, 3] = np.cos(theta)
    return m
