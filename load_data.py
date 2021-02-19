import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
class ReadDataset(data.Dataset):
    def __init__(self, imgs_path, labels_path, resize_size, n_classes):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.imgs_file = self.read_file(self.imgs_path) ##用来保存所有图片path
        self.labels_file = self.read_file(self.labels_path) ##class中定义的函数调用要加self
        self.resize_size = resize_size
        self.n_classes = n_classes
    
    def __getitem__(self, index):
        ## 生成实例后，调用要加上参数index
        ## 这个函数其实起到和Data.TensorDataset的功能，我们以前的数据是利用Data.TensorDataset(x,y)生成zip打包，变成类似于([x[0],y[0]],[x[1],y[1]],[x[2],y[2]]...)
        img = self.imgs_file[index] ## 第index个图片的path
        label = self.labels_file[index]
        img = Image.open(img)
        label = Image.open(label)  ##注意:这里label是img形式，对于不同的任务Label不一样
        img = self.img_resize(img, self.resize_size) ##定义img_resize函数在下面
        label = label.resize((self.resize_size,self.resize_size))
        label = np.array(label)
        '''
        ## 对label进行处理，转为(224x224xnum_class)
        seg_label = np.zeros((self.resize_size, self.resize_size, self.n_classes))
        for c in range(self.n_classes):
            seg_label[:,:,c] == (label[:,:] == c).astype(int)
            '''
        label = torch.tensor(label, dtype=torch.long) ##这样转换为tensor不会像totensor一样除255
        label = label.view(-1) ##因为loss用的交叉熵，会自动将label变为one-hot形式，所以将label变为一维
        return (img, label)  ##用__getitem__来代替TensorDataset必须要加入下面的def __len__，否则会提示无len

    def __len__(self):  
        return len(self.imgs_file)

    def read_file(self, path):
        file_list = os.listdir(path)
        file_path_list = [os.path.join(path,img) for img in file_list]
        file_path_list.sort()
        return file_path_list
    
    def img_resize(self,img, resize_size): ## 对图像进行resize并转换成tensor
        transform = transforms.Compose([
            transforms.Resize(size=(resize_size,resize_size)),
            transforms.ToTensor()  ## ToTensor会对图片进行正则化，即/255并且将(w,h,f)转换为(f,w,h)
        ])
        img = transform(img)
        return img
        
