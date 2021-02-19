import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
from model import Segnet
import torch.utils.data as Data
import argparse
from load_data import ReadDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## settings
parser = argparse.ArgumentParser(description="segnet pytorch-a simple implementation")
parser.add_argument('--resize_size', default=224)
parser.add_argument('--data_path', default='datasets/train/')
parser.add_argument('--test_path', default='datasets/test/')
parser.add_argument('--batch_size', default=1) ##GPU太小，batch_size先取1
args = parser.parse_args()

## parameters
n_classes = 2
LR = 0.0001
EPOCHS = 1

## train_data_loader 
imgs_file_path = args.data_path+'imgs/'
labels_file_path = args.data_path+'labels/'
my_data = ReadDataset(imgs_file_path, labels_file_path, args.resize_size, n_classes)
train_loader = Data.DataLoader(my_data, batch_size=args.batch_size, shuffle=True)

## test_data_loader
test_imgs_file_path = args.test_path+'imgs/'
test_labels_file_path = args.test_path+'labels/'
test_data = ReadDataset(test_imgs_file_path, test_labels_file_path, args.resize_size, n_classes)

## train
segnet = Segnet(3,n_classes).cuda()
optimizer = torch.optim.Adam(segnet.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        b_y = b_y.view(-1) ## 把batch_size也整在一起，都变成一维
        out_put = segnet(b_x)
        optimizer.zero_grad()
        loss = loss_func(out_put, b_y)  ##因为交叉熵会自动将label变为one-hot形式，所以b_x(-1,n_class),b_y是(-1)
        loss.backward()
        optimizer.step()
    if epoch%1 == 0:
        accuracy = 0
        for test_step, (test_x, test_y) in enumerate(test_data, 1): ## 1是使test_step从1开始计数
            test_x = Variable(test_x).cuda()
            with torch.no_grad():
                test_x = test_x.unsqueeze(0)
                test_output = segnet(test_x)
                test_output = torch.max(test_output, 1)[1].data.cpu().numpy()
                test_y = test_y.data.numpy()
                accuracy += sum(test_y == test_output)/len(test_y) ## 计算IoU
        accuracy /= test_step
        print("accuracy:", accuracy, "  | Loss:", loss.item())
torch.save(segnet,'logs/segnet.pth')

