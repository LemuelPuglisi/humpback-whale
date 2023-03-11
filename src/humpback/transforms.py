import torchvision as tv


def get_augmentation():
    gaussian = tv.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.5))
    return tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(), 
        tv.transforms.RandomGrayscale(p=0.4), 
        tv.transforms.RandomAffine(degrees=(-10, 10), translate=(0, 0.2), scale=(1, 1.1)), 
        tv.transforms.RandomApply([gaussian], p=0.5),
        tv.transforms.AugMix()
    ])

    
def get_data_loading(model_type='cnn'):
    shape = (256, 512) if model_type == 'cnn' else (384, 384)
    return tv.transforms.Compose([
        tv.transforms.Resize(shape),
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
