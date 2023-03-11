import os
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset


class HumpbackDataset(Dataset):
    
    def __init__(self, images_dir, csv_path, transforms=None):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.transforms = transforms
        self.annotations = pd.read_csv(csv_path, index_col=0)
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        filename, _, label = self.annotations.iloc[index]
        img = Image.open(os.path.join(self.images_dir, 'cropped_images', filename))
        if img.mode != 'RGB': img = img.convert('RGB')
        if self.transforms is not None: img = self.transforms(img)
        return img, label