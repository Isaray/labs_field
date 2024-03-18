"""
Refer to https://github.com/jiweibo/ImageNet
"""

import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    # traindir = os.path.join(root, 'ILSVRC2012_img_train')
    # valdir = os.path.join(root, 'ILSVRC2012_img_val')
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


if __name__ == '__main__':
    root = '/mnt/nfs/cl/datasets/imagenet-1k/'
    train_loader, val_loader = data_loader(root, 256, 8)
    
    for img, label in train_loader:
        print(img.shape, label.shape) # [256, 3, 224, 224], [256]
        break
    from IPython import embed
    embed()
    
    
    