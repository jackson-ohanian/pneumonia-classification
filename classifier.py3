import numpy as np
import torch
import math
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## const Crop size
IMAGE_W = 800
IMAGE_H = 800

################################################################################
## LungNet - custom nn module
## apply convulation layers to images
## apply relu to forward
################################################################################
class LungNet(nn.Module):

    def __init__(self):
        super(LungNet, self).__init__()
        self.conv1 = nn.Conv2d(3, IMAGE_W, IMAGE_W)
        self.conv2 = nn.Conv2d(IMAGE_W, 100, 1, 1)
        self.fc1 = nn.Linear(1, IMAGE_W)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        fc1 = self.fc1(conv2)
        print(fc1.shape)
        return fc1.view(10, 80000)

def loss_batch(model, loss_func, xb, yb, opt=None):
    target = (torch.FloatTensor(10).uniform_(0, 120).long())
    yb = yb.flatten()
    print(yb)
    print("bp - 1")
    resid_x = model(xb)
    print(resid_x.shape)
    loss = loss_func(resid_x, yb)
    if opt is not None:
        print("bp - 2")
        loss.backward()
        opt.step()
        print("bp - 3")
        opt.zero_grad()
    print("bp - 4")

    return loss.item(), len(xb)

def fit_and_validate(net, optimizer, loss_func, train, val, n_epochs, batch_size=1, exp_lr_gamma=0.95):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = exp_lr_gamma)
    #with torch.no_grad():
    #    # compute the mean loss on the training set at the beginning of iteration
    #    losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
    #    train_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
        # TODO compute the validation loss and store it in a list
    #    losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
    #    validation_epoch_loss = [np.sum(np.multiply(losses, nums)) / np.sum(nums)]
#
    for i in range(n_epochs):
        print("epoch - ", i)
        net.train() #put the net in train mode
        # TODO
        for X, Y in train_dl:
            print(Y.shape)
            loss_batch(net, loss_func, X, Y, optimizer)
            with torch.no_grad():
                net.eval() #put the net in evaluation mode
                # TODO compute the train and validation losses and store it in a list
                losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in train_dl])
                train_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))
                losses, nums = zip(*[loss_batch(net, loss_func, X, Y) for X, Y in val_dl])
                validation_epoch_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))
            scheduler.step()
    return train_epoch_loss, validation_epoch_loss


batch_size = 10


print("###### Begin Fitting #####")

### fit parameters
f_loss = torch.nn.CrossEntropyLoss()
epochs = 100
batch_size = 10
exp_lr_gamma = 0.95
lung_net = LungNet()
sgd = torch.optim.SGD(lung_net.parameters(), lr=0.005)

train_dl = torch.load("train_dl.pt")
val_dl = torch.load("val_dl.pt")

print("##################")
print("##### Params #####")
print("##### epochs < ", epochs, " >")
print("##### batch sz < ", batch_size, " >")
print("##### lr decay < ", exp_lr_gamma, " >")
print("##### opt type < ", "SGD", " >")
print("###################")


fit_and_validate(lung_net, sgd, f_loss, train_dl, val_dl, epochs, batch_size, exp_lr_gamma)
