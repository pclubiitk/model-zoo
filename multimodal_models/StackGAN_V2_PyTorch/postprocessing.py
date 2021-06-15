import torch
import torchvision.transforms as transforms

def postprocessing(tensor):
    tensor = ( tensor * 0.5 + 0.5 )
    tensor = tensor.mul_(255).type(torch.uint8)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor