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

class TwoModelClassifier(nn.Module):

    def __init__(self,model1,model2):
        super(TwoModelClassifier, self).__init__()

        self.model1=model1
        self.model2=model2

    def forward(self,x,inner_product=True):

        out1=self.model1(x,out_log_softmax=(not inner_product))
        out2=self.model2(x,out_log_softmax=(not inner_product))

        if(inner_product):
            soft_out1=F.softmax(out1,dim=1)
            soft_out2=F.softmax(out2,dim=1)

            #result=(soft_out1-soft_out2)
            inner_out=torch.sum(soft_out1*soft_out2,dim=1)
            result=torch.log(torch.stack([1-inner_out,inner_out],dim=1))

        else:
            result=(out1,out2)

        return result
