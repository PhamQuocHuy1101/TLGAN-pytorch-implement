from ast import iter_child_nodes
import torch
import torch.nn as nn

def init_weight(model):
    if isinstance(model, nn.Conv2d):
        nn.init.normal_(model.weight, 0, 0.02)
    if isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.bias, 0)

def compute_gradient(disc, img, f_img, esp):
    interpolate = esp * img + (1 - esp) * f_img
    disc_interpolate = disc(interpolate)
    gradient = torch.autograd.grad(
        outputs = disc_interpolate,
        inputs = interpolate,
        grad_outputs = torch.ones_like(disc_interpolate),
        retain_graph=True,
        create_graph=True
    )
    return gradient[0]

def gradient_penalty(grad):
    grad = grad.view(len(grad), -1)
    norm_grad = grad.norm(2, dim=1)
    return torch.mean(torch.pow(norm_grad - 1, 2))

def disc_loss(real, fake, grad):
    return torch.mean(real - fake + grad)