import torch

def map_compare(output, target):
    mu = output['mu']
    map_output = (mu > 0.5).type(torch.float)
    return torch.mean((map_output == target).type(torch.float))

def accuracy(output, target):
    mu = output['mu']
    map_output = (mu > 0.5).type(torch.float)
    return torch.mean((map_output == target).all(dim=1).type(torch.float))