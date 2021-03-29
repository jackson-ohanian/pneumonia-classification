import numpy as np
import torch
import math
import torchvision.transforms as transforms
import torchvision
import torch

################################################################################
## pneumonia-classification
## Jackson Ohanian
## github.io/jackson-ohanian
## jacksonohanian [at] gmail.com
################################################################################

################################################################################
## Using Data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
## With credit for compilation to https://data.mendeley.com/datasets/rscbjbr9sj/2
## Guangzhou Women and Children’s Medical Center
## and Paul Mooney http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5 (2018)
################################################################################

################################################################################
## Script - data_loaders
## run and done - change req. batch changes, transform changes, data changes
################################################################################

IMAGE_W = 800
IMAGE_H = 800

def load_training_data():
    ### The training images
    ### label - [0, normal] [1, pneumonia]
    data_path = 'chest_xray/train/'
    trans_list = [transforms.CenterCrop((IMAGE_W, IMAGE_H)), transforms.ToTensor()]
    trans_seq = transforms.Compose(trans_list)
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform = trans_seq)
    train_label = np.concatenate((np.ones([1341]), np.zeros([3875])))
    ### out - DL object containing images as Tensors
    return train_dataset, train_label

def load_validation_data():
    ### The training images
    ### label - [0, normal] [1, pneumonia]
    data_path = 'chest_xray/val/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor())
    train_label = np.concatenate((np.ones([1341]), np.zeros([3875])))
    ### out - DL object containing images as Tensors
    return train_dataset, train_label

def load_test_data():
    ### The testing images
    ### label - [0, normal] [1, pneumonia]
    data_path = 'chest_xray/test/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor())
    ### out - DL object containing images as Tensors
    return train_dataset




################################################################################
## Convert image dir datasets to dataloaders
## save raw for use - [train_dl val_dl test_dl]
## (Script) (Run on data change / batch change)
################################################################################

batch_size = 10

X_loader, y_loader = load_training_data()
X_validate_loader, drop = load_validation_data()

### default data sorted by directory, shuffle/avoid bias from model built this way
train_dl = torch.utils.data.DataLoader(X_loader, batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(X_validate_loader, batch_size)

torch.save(train_dl, "train_dl.pt")
torch.save(val_dl, "val_dl.pt")

### Memory approaches 13Gb
del X_loader
del X_validate_loader
print("saved - train_dl - val_dl")
