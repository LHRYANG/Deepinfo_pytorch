import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.first_encoder=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)

        )


        self.second_encoder=nn.Sequential(
            nn.Conv2d(256,256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.fc1=nn.Linear(256,256)
        self.fc2 =nn.Linear(256,256)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self,x):
        first_feature=self.first_encoder(x)
        #print(first_feature.shape)
        second_feature=self.second_encoder(first_feature)
        #print(second_feature.shape)
        output = F.max_pool2d(second_feature,4).view(64,256)
        #print(output.shape)
        z_mean=self.fc1(output)
        z_var=self.fc2(output)
        z=self.reparameterize(z_mean,z_var)
        return z,z_mean,z_var,first_feature


class GlobalDis(nn.Module):
    def __init__(self):
        super(GlobalDis, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        output=self.fc(x)
        return  output

class LocalDis(nn.Module):
    def __init__(self):
        super(LocalDis, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output=self.fc(x)
        return  output
