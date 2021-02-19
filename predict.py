import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

segnet = torch.load('logs/segnet.pth')
#segnet.eval()  ##eval()会使BN不起作用，正常predict是要加上eval，但是因为batch_size太小预测结果会受BN影响严重，我先不加，来测试网络是否正常
## predict
transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()  ## ToTensor会对图片进行正则化，即/255
        ])
test_img = '0.jpg'
test_img = Image.open(test_img)
test_img = transform(test_img)
test_img = test_img.unsqueeze(0)
test_img = Variable(test_img).cuda()
pred_img = segnet(test_img)
pred_img = pred_img.reshape(224,224,-1)
## 这里通道数是n_classes，选择大的
pred_img = torch.max(pred_img, 2)[1].data.cpu().numpy()
print(pred_img)
plt.imshow(pred_img)
plt.show()