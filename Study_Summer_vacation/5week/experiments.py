import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import argparse
def experiment(args):
    model = MLP(args.in_dim,args.out_dim,args.hid_dim,args.n_layer,args.act)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = args.lr,momentum = args.mm)

    for epoch in tqdm(range(args.epoch)):  # loop over the dataset multiple times
        # === Train === #
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.view(-1,3072)

            inputs = inputs.cuda()
            labels = labels.cuda()

            # print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # === validation === #
        correct = 0
        total = 0
        loss_arr = []
        val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images,labels = data
                images = images.view(-1,3072)
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                loss = criterion(outputs,labels)
                val_loss += loss.item()
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss = val_loss / len(valloader)
            val_acc = 100 * correct/total
            print('Epoch : {}, Train loss : {:.3f}, Val Loss : {:.3f} Val Accuracy : {:.3f}'.format(epoch,train_loss,val_loss,val_acc))
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images = images.view(-1,3072)
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(testloader)

    print('Accuracy : {}, val loss : {:.3f}'.format(100 * correct/total, val_loss))
    print('Finished Training')

if __name__ == '__main__':
    transform = T.Compose(
    [T.ToTensor(), # tensor 
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # -0.5~0.5 범위로 바꿔주고 , 다시 0.5로 나눠준다(채널별로)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = 4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset,batch_size = 4, shuffle=False,num_workers=2)

    # trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])

    valloader = torch.utils.data.DataLoader(valset,batch_size = 4, shuffle=False,num_workers=2)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layer', type=int,default=5)
    parser.add_argument('--in_dim', type=int,default=3072)
    parser.add_argument('--out_dim', type=int,default=100)
    parser.add_argument('--act', type=str,default='relu')
    parser.add_argument('--lr', type=float,default=0.001)
    parser.add_argument('--mm', type=float,default=0.9)
    parser.add_argument('--epoch', type=int,default=2)
    args = parser.parse_args()
    print(args)
    experiment(args)

