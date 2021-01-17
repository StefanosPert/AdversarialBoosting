from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import cw
from tqdm import tqdm

class PGD(object):

    def __init__(self,  eps=0.35, alpha=2 / 255, steps=400, min=0,max=1,random_start=False,device='cuda'):
        self.device=device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.min=min
        self.max=max
        self.random_start = random_start

    def forward(self,model, images, labels):
        r"""
        Overridden.
        """
        self.model=model
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #labels = self._transform_label(images, labels)    
        loss = nn.NLLLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            #print(outputs)
            
            cost = -loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.min, max=self.max).detach()
        #print(outputs)
        return adv_images
