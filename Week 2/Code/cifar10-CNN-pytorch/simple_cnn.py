import torch
import torchvision
import torchvision.transforms as transforms
# Image Transform

#数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
) # Compose的主要作用是将多个变换组合在一起，可以实现同时变换
# ToTensor 功能：将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 
# Normalize 分别传入三通道的mean和std

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')



import torch.nn as nn
import torch.nn.functional as F

#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(net)

import torch.optim as optim

#定义损失函数和优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


#训练
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0): # i是序列脚标，data是具体数据
        inputs, labels = data

        optimizer.zero_grad()

        # 前向传播，计算loss，反向传播，权重更新
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = running_loss+loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print('Finished Training')

#存储训练后的模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
