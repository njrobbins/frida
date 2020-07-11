# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:17:15 2020

@author: A
"""

# !unzip -q "dataset.zip"

#%load_ext autoreload
#%autoreload 2

root = ',/'

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms

from PIL import Image

#%matplotlib inline



data_dir = 'C:/Users/A/Desktop/fall/Fall-Detection-PyTorch/dataset/dataset'
print(data_dir)
batch_size = 32

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # convert 1 to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Train dataset and dataloader
train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_sizes = len(train_dataset)
class_names = train_dataset.classes

# Validation dataset and dataloader
val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

val_sizes = len(val_dataset)


class_names


train_sizes, val_sizes

len(train_dataset)

class FDNet(nn.Module):
    def __init__(self, out_features=2):
        super(FDNet, self).__init__()
        mnet = models.mobilenet_v2(pretrained=True)
        for name, param in mnet.named_parameters():
            if("bn" not in name):
                param.requires_grad_(False)

        # Parameters of newly constructed modules have requires_grad=True by default
        in_features = mnet.classifier[1].in_features
        mnet.classifier = nn.Sequential(
                                nn.Dropout(p=0.2, inplace=False),
                                nn.Linear(in_features,500),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(500, out_features))
        self.mnet = mnet

    def forward(self, images):
        features = self.mnet(images)

        return features




# Initialize the model.
model = FDNet()

# Move models to GPU if CUDA is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def train(model, optimizer, loss_fn, train_loader, val_loader, start_epoch = 0, epochs=20, device="cpu"):
    model.to(device)
    for epoch in range(start_epoch, start_epoch+epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))


# Initialize the model.
model = FDNet()

# Define the loss function.
criterion = nn.CrossEntropyLoss()

# Define the optimizer.
params = model.mnet.classifier.parameters()
optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

epochs = 20


#%%time
train(model, optimizer, criterion, train_dataloader, val_dataloader, epochs=epochs, device=device)


os.makedirs(f'{root}/train_model', exist_ok=True)
PATH = f'{root}/train_model/fdnet.pt'
torch.save(model.state_dict(), PATH)


# Test dataset and dataloader
test_dataset = datasets.ImageFolder(f'{data_dir}/test', transform)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

test_sizes = len(test_dataset)



test_sizes

outputs_np = np.empty((0), dtype=int)
targets_np = np.empty((0), dtype=int)

model.eval()
num_correct = 0
num_examples = 0
for batch in test_dataloader:
    inputs, targets = batch
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputs = torch.max(F.softmax(outputs, dim=1), dim=1)[1]
    targets = targets.to(device)
    correct = torch.eq(outputs, targets).view(-1)
    num_correct += torch.sum(correct).item()
    num_examples += correct.shape[0]

    outputs_np = np.concatenate([outputs_np, outputs.cpu().numpy()], axis=None)
    targets_np = np.concatenate([targets_np, targets.cpu().numpy()], axis=None)

print(f'Accuracy: {num_correct / num_examples:.2f}')
print(num_correct, num_examples)

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(targets_np, outputs_np)

cf


tn, fp, fn, tp = cf.ravel()

tn, fp, fn, tp