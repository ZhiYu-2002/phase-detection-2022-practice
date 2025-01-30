from cProfile import label
import os
import io
import h5py
import time
import numpy as np
import pandas as pd
from PIL import Image
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.pooling import AvgPool2d
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from skimage import io
from pathlib import Path
from os.path import splitext
from os import listdir

class MyDataset(Dataset):
    def __init__(self, imgfile_path, txtfile_path, transform):

        label = []
        idx = []
        img = []
        if os.path.isdir(imgfile_path):
            imgs = os.listdir(imgfile_path)#图片名，列表
        
        txtdata = open(txtfile_path, 'r')
        for line in txtdata:
            words = line.split()#以tab按行切割txt的idx, label
            label.append(words[1])
            idx.append(words[0])
        
        for filename in imgs:#按idx顺序排好图片名并push进img列表
            itsname = filename.split('.')
            firstname = itsname[0]
            ininame = firstname[0:5]
        for id in idx:
            newname = os.path.join(imgfile_path, ininame + id + '.jpg')
            img.append(newname)
            
        self.imgs = imgs
        self.img = img
        self.label = label
        self.idx = idx
        self.transform = transform
    
    def __getitem__(self, idx):
        fn = self.img[idx]
        labels = self.label[idx]
        image = Image.open(fn)
        image = self.transform(image)
        return image, labels
    
    def __len__(self):
        return len(self.img)#图片list长

def train_model(model, trainloader, criterion, optimizer, lr_scheduler, num_epochs = 30):
    
    running_Loss = 0
    running_Acc = 0

    train_loss_list = []
    train_acc_list = []

    for epoch in range(num_epochs):
        
        print("Epoch{epoch} starting")
        loss_train = 0

        for data in trainloader: 

            optimizer.zero_grad()

            inputs, labels = data
            inputs = inputs.cuda()

            output = model(inputs)
            multi = nn.Sigmoid()
            output = multi(output)
            output.resize(1, 7)

            labels = " ".join(labels)
            labels = int(labels)
            labels = torch.tensor([labels])
            labels = labels.cuda()

            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()
            running_Loss += loss.item()
        running_Loss = running_Loss / len(trainloader)
        print("Epoch{epoch}:finished")
        print(loss.item())
        #train_acc_list.append(running_Acc.item())
        train_loss_list.append(running_Loss)
    #Acc = {}
    Loss = {}
    #Acc['train_acc'] = train_acc_list
    Loss['train_loss'] = train_loss_list
    #return  running_Acc, running_Loss, Acc, Loss
    return running_Loss, Loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=0, dilation =1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )


        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1000, out_features=7, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def plotMyPicture(num_epochs, Loss):

    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1, num_epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train'], loc = 'upper left')
    plt.show()

if __name__=='__main__':

    num_epochs = 30

    t = time.time()

    #trainingLoaderSet = []

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    datasetPath = "D://2021autumn//totaldataset//first dataset video//" #图片
    labelPath = "D://2020autumn//myvideo-phase.txt" #标签

    train_data = MyDataset(imgfile_path = datasetPath, txtfile_path = labelPath, transform = transform)
    trainloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last = True)
    
    #trainingLoaderSet.append(trainloader)
    print("Load all the dataset costs %d minutes %d seconds"%((time.time()-t)/60, (time.time()-t)%60))


    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    #model = torch.load(Net())
    model.train()
    print("Is the model in training mode?", model.training)
    running_Loss, Loss = train_model(model, trainloader, criterion, optimizer, lr_scheduler = optimizer_scheduler, num_epochs = 30)
    print(running_Loss)
    plotMyPicture(num_epochs, Loss)