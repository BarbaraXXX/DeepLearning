# train.py文件
# 包含main函数，其中内容又包括数据集加载， 训练参数设置， 训练函数设置， 测试函数设置等
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import os

from model import Net

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():

    # 超参数设定
    epochs = 10   # 训练轮数
    lr = 0.001    # 学习率
    momentum = 0.9  # stochastic gradient decent下降算法动量
    batch_size = 64  # 每个batch的数量
    num_workers = 8     # Dataloader采用多个进程读取并行数据

    # 数据集准备
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='dataset/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers,)
    testset = torchvision.datasets.CIFAR10(root='dataset/cifar10', train=True,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on {device}')

    # 模型加载
    net = Net().to(device)

    # 训练模型

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # 梯度下降法优化参数
    criterion = nn.CrossEntropyLoss()
    # 损失计算采用交叉熵损失

    # 开始训练
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 训练
        net.train()     # 设置模式为训练模式
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # 输入转移到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练集loss
            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'training loss in epoch{ epoch + 1} {i + 1}th img is '
                      f'{running_loss / 2000:.2f}')
                running_loss = 0.0

            # 计算训练集准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'the {epoch + 1}epoch loss is {epoch_loss:.2f}')
        print(f'the acc of this train epoch is:{100 * correct / total:.2f}%')
        total = 0
        correct = 0
        # 测试
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data

                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'the acc of this test epoch is:{100 * correct / total:.2f}%')

    # 训练完成，查看图像
    end_time = time.time()
    print(f'the training cost {end_time - start_time}sec')

    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # imgshow(torchvision.utils.make_grid(images.cpu()))
    # print('groundtruth:', ''.join('%5s'% classes[labels[j]] for j in range(4)))












def imgshow(image):
    image = image / 2 + 0.5
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def use_gpu():
    # wrong
    if torch.cuda.is_available():
        print(f'training on cuda:0\n')
        return torch.device('cuda')
    else:
        print(f'training on cpu\n')
        return torch.device('cpu')


if __name__ == "__main__":
    main()

    # 尝试在GPU上训练
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # x = torch.randn((1, 1, 10, 10), requires_grad=True)
    # x.to(device)
    # print(device)
    # print(torch.cuda.is_available())
    #
    # print(x.device)

