from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from pgd import PGD
from main_mnist import Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from two_model_clasifier import TwoModelClassifier
from Trainer import Trainer
import cw
def regular_adversary_checker(model,data,target,adversary,device='cuda'):
    adversarial_examples = adversary(model, data, target)
    out=model(adversarial_examples.to(device))

    _,labels=torch.max(out.detach(),dim=1)
    index_mask=labels!=target

    return index_mask, adversarial_examples

def difference_checker(model,data,target,adversary,device='cuda'):
    new_target=torch.ones(data.shape[0],device=device)
    adversarial_examples = adversary(model, data, new_target)

    out1,out2=model(adversarial_examples.to(device),inner_product=False)

    _,labels1=torch.max(out1.detach(),dim=1)
    _,labels2=torch.max(out2.detach(),dim=1)
    index_mask=labels1!=labels2

    return index_mask
def create_adv(model,adversary,loader,adversary_checker,logdir='./adv_check',device='cuda'):
    image_list=[]
    label_list=[]
    model.eval()
    cnt=0
    for data, target in loader:
        data, target=data.to(device), target.to(device)

        #adversarial_examples=adversary(model,data,target)

        indexes,_=adversary_checker(model,data,target,adversary)
        if torch.sum(indexes.float())>0:
            adv_data=data[indexes].to('cpu')
            adv_labels=target[indexes].to('cpu')
            image_list.append(adv_data)
            label_list.append(adv_labels)
            print("Average adversarial in batch ="+str(torch.mean(indexes.float()).item()))
        cnt+=1
        #if(cnt>10):
        #    break
    adversarial_tensor=torch.cat(image_list,dim=0)
    adversarial_length=adversarial_tensor.shape[0]
    del adversarial_tensor

    total_samples=0
    for data, target in loader:
        data, target=data.to(device), target.to(device)
        total_samples+=data.shape[0]
        image_list.append(data.to('cpu'))
        label_list.append(target.to('cpu'))
        if(total_samples>adversarial_length):
            break

    save_data={'data_tensor': torch.cat(image_list,dim=0),
                'label_tensor': torch.cat(label_list,dim=0)}
    torch.save(save_data,logdir)

if __name__== '__main__':
    parser=argparse.ArgumentParser(description='Create Adversarial Examples')
    parser.add_argument('--saved_model', type=str,default='./mnist.pt')
    parser.add_argument('--logdir', type=str,default='./dataset.pt')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--set_second_model',type=str,default='None')
    args=parser.parse_args()
    device='cuda'
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    train_dataset = datasets.MNIST('./data', train=True,
                              transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size)

    input_box=(-0.1307/0.3081,(1-0.1307)/0.3081)
    '''
    adversary = cw.L2Adversary(targeted=False, confidence=0.0, search_steps=10, c_range=(1e-3, 1e10), box=input_box,
                                   optimizer_lr=5e-3)
    '''
    adversary=PGD(min=input_box[0],max=input_box[1],random_start=True).forward

 

    model=Net()
    model=model.to(device)

    model.load_state_dict(torch.load(args.saved_model))

    if args.set_second_model=='None':
        adversary_checker=regular_adversary_checker
    else:
        model2=Net()
        model2=model2.to(device)
        model2.load_state_dict(torch.load(args.saved_model))

        new_model=TwoModelClassifier(model,model2)

        adversary_checker=difference_checker

    create_adv(model,adversary,train_loader,adversary_checker,logdir=args.logdir)

