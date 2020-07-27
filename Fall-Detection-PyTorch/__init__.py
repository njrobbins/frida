# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:09:44 2020

@author: A
"""
import torch
import torch.nn as nn
from torchvision import models

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




model.load_state_dict(torch.load("C:/Users/A/Desktop/main-fri/frida/Fall-Detection-PyTorch/train_model/fdnet.pt", map_location=torch.device('cpu')))
print("we eval?")
model.eval()
input = torch.randn(32, 1, 128, 128, requires_grad=True)
output = model(input)
torch_out = torch.onnx._export(model, input, "model.onnx", export_params=True, do_constant_folding=True)

for parems in model.parameters():
    print(parems)