import numpy as np
import torch
from torch.utils.data import DataLoader


def images_to_embeddings(model, dataset, device='cpu', batch_size=16):
    model.eval()
    with torch.no_grad():
        embeddings = []
        data_loader = DataLoader(dataset, batch_size, False)
        for images, _ in data_loader:
            images = images.to(device)
            res = model(images).cpu().numpy()
            embeddings += list(res)
        return np.stack(embeddings, axis=0)
    
    
def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True