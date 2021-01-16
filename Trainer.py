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

class Trainer(object):
    def __init__(self, model,device,train_loader,test_loader,num_epochs,start_adversarial=10,logdir='./mnist.pt'):
        self.device=device
        self.model=model.to(device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.num_epochs=num_epochs
        self.start_adversarial=start_adversarial
        input_box = (-0.1307 / 0.3081, (1 - 0.1307) / 0.3081)
        self.adversary = cw.L2Adversary(targeted=False, confidence=0.0, search_steps=10, c_range=(1e-3, 1e10), box=input_box,
                                   optimizer_lr=5e-3)
        self.logdir=logdir


    def train(self,args, model, device, train_loader, optimizer, epoch, adversarial=False, adversary=None):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            if adversarial:
                if(batch_idx>len(train_loader)*0.1):
                  break
                data = adversary(model, data, target, to_numpy=False)
                data = data.to(device)
                #print("Adversarial Training Batch Id=" + str(batch_idx))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break

    def test(self,model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def run_train(self,args):
        self.model=self.model.to(self.device)

        optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            if epoch > self.start_adversarial:
                self.train(args, self.model, self.device, self.train_loader, optimizer, epoch, adversarial=True, adversary=self.adversary)
            else:
                self.train(args, self.model, self.device, self.train_loader, optimizer, epoch)
            self.test(self.model, self.device, self.test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(self.model.state_dict(), self.logdir)