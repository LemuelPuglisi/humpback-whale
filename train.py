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
from humpback.utils import unfreeze_module, freeze_module


random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--grad-acc-size', type=int, default=8)
parser.add_argument('--grad-acc-step', type=int, default=8)
parser.add_argument('--emb-size', type=int, default=2048)
parser.add_argument('--backbone', type=str, default='cnn')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--eta-min', type=float, default=1e-7)
args = parser.parse_args()

device      = 'cuda' if torch.cuda.is_available() else 'cpu'
images_dir  = args.images_dir
epochs      = args.epochs
batch_size  = args.grad_acc_size
accum_iter  = args.grad_acc_step
emb_size    = args.emb_size
model_type  = args.backbone
lr          = args.lr
wd          = args.wd
eta_min     = args.eta_min

print(f'> running on {device}')

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
    
    if (epoch + 1) == 50: unfreeze_module(model)
         
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