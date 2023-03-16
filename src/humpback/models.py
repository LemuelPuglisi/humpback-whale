from torch import nn
from torchvision import models

from humpback.utils import freeze_module, unfreeze_module


def get_resnet(embedding_size=2048):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=2048, out_features=embedding_size)    
    freeze_module(model)
    unfreeze_module(model.fc)
    return model
    

def get_vit(embedding_size=2048):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    model.heads = nn.Linear(in_features=768, out_features=embedding_size) # type: ignore
    freeze_module(model)
    unfreeze_module(model.heads)
    return model