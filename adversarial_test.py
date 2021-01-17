from main_mnist import Net
import cw
from pgd import PGD
from torchvision import datasets, transforms
import torch
import argparse
from ensemble_classifier import EnsembleClassifier
import matplotlib.pyplot as plt
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Create Adversarial Examples')
    parser.add_argument('--saved_model', type=str,default='./mnist.pt')
    parser.add_argument('--saved_model2', type=str,default='None')
    parser.add_argument('--saved_model3', type=str, default='None' )
    args=parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size=16
    saved_model = args.saved_model

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    test_dataset = datasets.MNIST('./data', train=False,
                              transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=batch_size)

    mnist_net=Net()
    mnist_net=mnist_net.to(device)

    mnist_net.load_state_dict(torch.load(saved_model))
    if(args.saved_model2!='None' and args.saved_model3!='None'):
        model2=Net()
        model3=Net()
        model2, model3= model2.to(device), model3.to(device)

        new_ensemble=EnsembleClassifier(mnist_net,model2,model3)

        mnist_net=new_ensemble

    #input, target=next(iter(test_loader))
    adv_percentage=0
    percentage=0
    mnist_net.eval()
    for batch_id, (input, target) in enumerate(test_loader):
        input=input.to(device).float()
        target=target.to(device)

        #print(input.shape)
        out=mnist_net(input)
        max_conf, labels=torch.max(out.detach(),dim=1)
        percentage+=torch.mean((labels!=target).float()).item()
        #print(labels)


        #print("Max")
        max=input[:].max()
        min=input[:].min()

        input_box=(-0.1307/0.3081,(1-0.1307)/0.3081)
        '''
        adversary = cw.L2Adversary(targeted=False, confidence=0.0, search_steps=10, c_range=(1e-3, 1e10), box=input_box,
                                   optimizer_lr=5e-3)
        '''
        adversary=PGD(min=input_box[0],max=input_box[1],random_start=True).forward
    
        adversarial_examples=adversary(mnist_net,input,target)
        '''
        fig, ax=plt.subplots(4,int(batch_size/4))

        #print(labels)
        for i in range(4):
            for j in range(int(batch_size/4)):
                ax[i,j].imshow(input[i*4+j,0,:,:].to('cpu'),cmap='gray')
                ax[i,j].set_title('Cat: '+str(labels[i*4+j].item()))
                ax[i,j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        plt.show()
        '''
        out=mnist_net(adversarial_examples.to(device))

        max_conf, labels = torch.max(out.detach(), dim=1)
        #fig, ax = plt.subplots(4, int(batch_size / 4))
        adv_percentage+=torch.mean((labels!=target).float()).item()
        #print(labels)
        print("Adversarial Percentage="+str(adv_percentage/(batch_id+1))+' Percentage on Original Distribution='+str(percentage/(batch_id+1)))
        
        '''
        for i in range(4):
            for j in range(int(batch_size/4)):
                ax[i,j].imshow(adversarial_examples[i*4+j,0,:,:].to('cpu'),cmap='gray')
                ax[i,j].set_title('Cat: '+str(labels[i*4+j].item()))
                ax[i,j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        plt.show()
        '''
