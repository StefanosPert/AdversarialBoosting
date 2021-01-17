import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Trainer import Trainer
import cw
from torch.utils.data import TensorDataset


class EnsembleClassifier(nn.Module):

    def __init__(self, model1, model2, model3):
        super(EnsembleClassifier, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3= model3

    def forward(self, x):

        out1 = self.model1(x, out_log_softmax=False)
        out2 = self.model2(x, out_log_softmax=False)
        out3 = self.model3(x, out_log_softmax=False)

        soft_out1 = F.softmax(out1, dim=1)
        soft_out2 = F.softmax(out2, dim=1)
        soft_out3= F.softmax(out3,dim=1)

        result=torch.log((soft_out1+soft_out2+soft_out3)/3)

        return result