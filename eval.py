import argparse

import torch
from humpback.dataset import HumpbackDataset
from humpback.models import get_resnet, get_vit
from humpback.transforms import get_data_loading
from humpback.metrics import evaluate_model_nw

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--model-ckpt', type=str, required=True)
parser.add_argument('--model-type', type=str, required=True)
parser.add_argument('--emb-size', type=int, default=2048)
parser.add_argument('--nw-threshold', type=float, default=0.55)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
images_dir = args.images_dir
model_type = args.model_type
model_ckpt = args.model_ckpt
new_whales_th = args.nw_threshold
emb_size = args.emb_size

load = get_data_loading(model_type)
validset = HumpbackDataset(images_dir=images_dir, csv_path='annotations/valid_nw.csv', transforms=load)
testset  = HumpbackDataset(images_dir=images_dir, csv_path='annotations/test_nw.csv', transforms=load)

model = get_resnet(emb_size) if model_type == 'cnn' else get_vit()
model.load_state_dict(torch.load(model_ckpt))
model = model.to(device).eval()

vmap = evaluate_model_nw(model, validset, device, emb_size, new_whales_th)
print('validation mAP@5: ', vmap)

tmap = evaluate_model_nw(model, testset, device, emb_size, new_whales_th)
print('test mAP@5: ', tmap)