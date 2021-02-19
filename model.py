import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class Segnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Segnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ## encoder 
        #  stage_1
        self.conv1_1 = nn.Sequential(  ## (224,224,3) ->(224,224,64)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(  ## (224,224,64) ->(224,224,64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )
        self.pool_1 = nn.MaxPool2d(2)  ##(224,224,64) ->(112,112,64)
        #  stage_2
        self.conv2_1 = nn.Sequential(  ## (112,112,64) ->(112,112,128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(  ## (112,112,128) ->(112,112,128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU()
        )
        self.pool_2 = nn.MaxPool2d(2)  ## (112,112,128) ->(56,56,128)
        #  stage_3
        self.conv3_1 = nn.Sequential(  ## (112,112,128) ->(56,56,256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(  ## (56,56,256) ->(56,56,256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(  ## (56,56,256) ->(56,56,256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        self.pool_3 = nn.MaxPool2d(2)  ## (56,56,256) ->(28,28,256)
        #  stage_4
        self.conv4_1 = nn.Sequential(  ## (28,28,256) ->(28,28,512)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(  ## (28,28,512) ->(28,28,512)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU()
        )
        self.conv4_3 = nn.Sequential(  ## (28,28,512) ->(28,28,512)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.ReLU()
        )
        self.pool_4 = nn.MaxPool2d(2)  ## (28,28,512) ->(14,14,512)
        ##  decoder
        #   stage_4
        self.upsample_4 = nn.Upsample(2)  ## (14,14,512) ->(28,28,512)
        self.d_conv4_3 = nn.Sequential(  ## (28,28,512) ->(28,28,256)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv4_2 = nn.Sequential(  ## (28,28,256) ->(28,28,256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv4_1 = nn.Sequential(  ## (28,28,256) ->(28,28,256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU()
        )
        #   stage_3
        self.upsample_3 = nn.Upsample(2)  ## (28,28,256) ->(56,56,256)
        self.d_conv3_3 = nn.Sequential(  ## (56,56,256) ->(56,56,128)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv3_2 = nn.Sequential(  ## (56,56,128) ->(56,56,128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv3_1 = nn.Sequential(  ## (56,56,128) ->(56,56,128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU()
        )
        #   stage_2
        self.upsample_2 = nn.Upsample(2)  ## (56,56,128) ->(112,112,128)
        self.d_conv2_2 = nn.Sequential(  ## (112,112,128) ->(112,112,64)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv2_1 = nn.Sequential(  ## (112,112,64) ->(112,112,64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )
        #   stage_1
        self.upsample_1 = nn.Upsample(2)  ## (112,112,64) ->(224,224,64)
        self.d_conv1_2 = nn.Sequential(  ## (224,224,64) ->(224,224,64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )
        self.d_conv1_1 = nn.Sequential(  ## (224,224,64) ->(224,224,out_channels)
            nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        ## encoder
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        ## decoder
        x = self.d_conv4_3(x)
        x = self.d_conv4_2(x)
        x = self.d_conv4_1(x)
        x = self.d_conv3_3(x)
        x = self.d_conv3_2(x)
        x = self.d_conv3_1(x)
        x = self.d_conv2_2(x)
        x = self.d_conv2_1(x)
        x = self.d_conv1_2(x)
        x = self.d_conv1_1(x)
          
        x = x.permute(0,2,3,1).contiguous()  ## 变换前0是batch_size，1是通道数，2,3是w,h
        x = x.view(-1, self.out_channels)
        return x


        