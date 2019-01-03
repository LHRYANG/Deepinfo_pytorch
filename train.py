from opts import args

from model import Encoder
from model import GlobalDis
from model import LocalDis

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder=Encoder().to(device)
globaldis=GlobalDis().to(device)
localdis=LocalDis().to(device)

optimizer = optim.Adam([{'params': encoder.parameters()},
                        {'params': globaldis.parameters()},
                        {'params': localdis.parameters()}], lr=1e-3)

def shuffle(x):
    idx=torch.randperm(len(x))
    change_x=x[idx]

    return change_x


def loss_function(mu, logvar,z_z_1_score,z_z_2_score,z_f_1_score,z_f_2_score):
    GIL= - torch.sum(torch.log(z_z_1_score + 1e-6) + torch.log(1 - z_z_2_score + 1e-6))
    LIL= - torch.sum(torch.log(z_f_1_score + 1e-6) + torch.log(1 - z_f_2_score + 1e-6))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return args.alpha*GIL + args.beta*LIL + args.gamma*KLD


def train(epoch):
    train_loss=0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        inputs, labels = data
        if(len(labels)<args.batch_size):
            break
        z,z_mean,z_var,first_feature=encoder(inputs.to(device))

        change_z=shuffle(z)

        z_z_1=torch.cat((z,z),dim=1)
        z_z_2=torch.cat((z,change_z),dim=1)
        first_feature = torch.transpose(torch.transpose(first_feature, 1, 2), 2, 3)
        feature_map_shuffle = shuffle(first_feature)

        #print(feature_map_shuffle.shape)
        z_samples_map= z.repeat(1,4*4).view(z.shape[0],4,4,z.shape[-1])
        #print(z_samples_map.shape)
        z_f_1=torch.cat((z_samples_map,first_feature),dim=-1)
        z_f_2=torch.cat((z_samples_map,feature_map_shuffle),dim=-1)

        z_z_1_score=globaldis(z_z_1)
        z_z_2_score=globaldis(z_z_2)

        z_f_1_score=localdis(z_f_1)
        z_f_2_score=localdis(z_f_2)



        loss=loss_function(z_mean,z_var,z_z_1_score,z_z_2_score,z_f_1_score,z_f_2_score)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if i % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(labels), len(trainloader.dataset),
                       100. * i / len(trainloader),
                       loss.item() / len(labels)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainloader.dataset)))

for epoch in range(args.epochs):
    train(epoch)


test_feature=[]
image=[]
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data


        if(len(labels)<64):
            break
        z, z_mean, z_var, first_feature = encoder(inputs.to(device))
        test_feature.append(z.cpu().numpy())
        image.append(inputs.cpu().numpy())


test_feature=np.array(test_feature)
test_feature=test_feature.reshape(-1,256)

image=np.array(image).reshape(-1,3,32,32)
print(image.shape)


def imshow(img,name):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("./results/"+name+".png")

def getSimilar(n,topk):

    for i in range(n):


        one =np.random.randint(len(test_feature))

        one_feature=test_feature[one]

        similarity=test_feature.dot(one_feature)

        indx=similarity.argsort()[-topk:][::-1]

        results=image[indx]

        one_image=image[one].reshape(-1,3,32,32)
        result=np.append(results,one_image,axis=0)
        images=torch.tensor(result)
        imshow(torchvision.utils.make_grid(images),str(i))


getSimilar(10,15)