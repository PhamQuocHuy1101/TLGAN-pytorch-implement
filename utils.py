import torch.nn as nn

def init_weight(model):
    if isinstance(model, nn.Conv2d):
        nn.init.normal_(model.weight, 0, 0.02)
    if isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.bias, 0)