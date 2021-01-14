from main_mnist import Net
import cw

from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size=16
    saved_model = 'mnist_cnn.pt'

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    test_dataset = datasets.MNIST('./data', train=True,
                              transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    mnist_net=Net()
    mnist_net=mnist_net.to(device)

    mnist_net.load_state_dict(torch.load(saved_model))

    input, target=next(iter(test_loader))

    mnist_net.eval()
    input=input.to(device).float()
    target=target.to(device)

    print(input.shape)
    out=mnist_net(input)
    max_conf, labels=torch.max(out.detach(),dim=1)
    print("Max")
    max=input[:].max()
    min=input[:].min()

    input_box=(-0.1307/0.3081,(1-0.1307)/0.3081)
    adversary=cw.L2Adversary(targeted=False,confidence=0.0,search_steps=10,c_range=(1e-3, 1e10),box=input_box,optimizer_lr=5e-4)

    adversarial_examples=adversary(mnist_net,input,target,to_numpy=False)

    fig, ax=plt.subplots(4,int(batch_size/4))

    print(labels)
    for i in range(4):
        for j in range(int(batch_size/4)):
            ax[i,j].imshow(input[i*4+j,0,:,:].to('cpu'),cmap='gray')
            ax[i,j].set_title('Cat: '+str(labels[i*4+j].item()))
            ax[i,j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    plt.show()

    out=mnist_net(adversarial_examples.to(device))

    max_conf, labels = torch.max(out.detach(), dim=1)
    fig, ax = plt.subplots(4, int(batch_size / 4))
    percentage=torch.mean((labels!=target).float())
    print(labels)
    print("Adversarial Percentage="+str(percentage))
    for i in range(4):
        for j in range(int(batch_size/4)):
            ax[i,j].imshow(adversarial_examples[i*4+j,0,:,:].to('cpu'),cmap='gray')
            ax[i,j].set_title('Cat: '+str(labels[i*4+j].item()))
            ax[i,j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    plt.show()