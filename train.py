import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

from humpback.transforms import get_augmentation, get_data_loading
from humpback.dataset import HumpbackDataset
from humpback.models import get_resnet, get_vit
from humpback.loss import CosFaceLoss
from humpback.metrics import evaluate_model


random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

device      = 'cuda'
epochs      = 120
batch_size  = 8
accum_iter  = 8
emb_size    = 2048
model_type  = 'cnn'
images_dir  = '/data/research-data/humpback-whale/humpback-whale-identification' 
lr          = 1e-3
wd          = 1e-5
eta_min     = 1e-7

augm = get_augmentation()
load = get_data_loading(model_type)
augm_load = transforms.Compose([augm, load])

trainset = HumpbackDataset(images_dir=images_dir, csv_path='annotations/train.csv', transforms=augm_load)
validset = HumpbackDataset(images_dir=images_dir, csv_path='annotations/valid.csv', transforms=load)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

n_whales = len(trainset.annotations.label.unique())
model = get_resnet(emb_size) if model_type == 'cnn' else get_vit()
criterion = CosFaceLoss(emb_size, n_whales)

param_groups = [
    { 'params': model.parameters(),     'lr': lr, 'weight_decay': wd }, 
    { 'params': criterion.parameters(), 'lr': lr, 'weight_decay': wd }
]

optimizer = AdamW(param_groups, lr=lr, weight_decay=wd)
scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=epochs)
writer = SummaryWriter('logs')

criterion = criterion.to(device)
model = model.to(device)
model.train()

for epoch in range(epochs):
    
    losses = []
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        images = images.to(device)
        labels = labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels) / float(accum_iter)
        losses.append(loss.item())
        
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/ep-{(epoch+1)}.ckpt')
            torch.save(criterion.state_dict(), f'checkpoints/ep-{(epoch+1)}-crit.ckpt')
        
    scheduler.step()
    
    # Logging train loss
    avg_epoch_loss = sum(losses) / len(losses)
    print(f'epoch loss:', avg_epoch_loss)
    writer.add_scalar(f'epoch_loss/train', avg_epoch_loss, global_step=epoch)
    
    # Logging valid mAP
    print('testing...')
    model.eval()
    map5 = evaluate_model(model, validset, 'cuda', embedding_dim=emb_size)
    writer.add_scalar(f'mAP@5/valid', map5, global_step=epoch)

    print('-' * 50)

torch.save(model.state_dict(), 'checkpoints/trained_model.ckpt')
torch.save(criterion.state_dict(), 'checkpoints/trained_criterion.ckpt')