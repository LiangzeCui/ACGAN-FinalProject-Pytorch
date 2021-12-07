import torch
import torch.nn as nn

class generator(nn.Module):

    #generator model
    def __init__(self,in_channels):
        super(generator,self).__init__()
        self.fc1=nn.Linear(in_channels,384)

        self.t1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=384,out_channels=192,kernel_size=(4,4),stride=1,padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.t2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=192,out_channels=96,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.t3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=96,out_channels=48,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.t4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=48,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )
    
    def forward(self,x):
    	x=x.view(-1,110)
    	x=self.fc1(x)
    	x=x.view(-1,384,1,1)
    	x=self.t1(x)
    	x=self.t2(x)
    	x=self.t3(x)
    	x=self.t4(x)
    	return x #output of generator
    
class discriminator(nn.Module):
    
    def __init__(self,classes=10):
        #we have 10 classes in the CIFAR dataset with 6000 images per class.
        super(discriminator,self).__init__()
        self.c1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.c6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
            )
        self.fc_source=nn.Linear(4*4*512,1)
        self.fc_class=nn.Linear(4*4*512,classes)
        self.sig=nn.Sigmoid()
        self.soft=nn.Softmax()

    def forward(self,x):

        x=self.c1(x)
        x=self.c2(x)
        x=self.c3(x)
        x=self.c4(x)
        x=self.c5(x)
        x=self.c6(x)
        x=x.view(-1,4*4*512)
        rf=self.sig(self.fc_source(x))#checks source of the data---i.e.--data generated(fake) or from training set(real)
        c=self.soft(self.fc_class(x))#checks class(label) of data--i.e. to which label the data belongs in the CIFAR10 dataset
        
        return rf,c 