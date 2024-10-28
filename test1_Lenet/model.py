import torch
import torch.nn as nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        # 通常采用CrossEntropyLoss时内部自动应用softmax,不需要添加
        # x = F.softmax(x, dim=1)

        return x

    def init_weights(self):
        # mode决定计算权重标准差时使用的基数， fan_in 根据输入神经元数量、fan_out 根据输出神经元数量
        # 对ReLU激活函数，适用fan_out。
        # nonlinearity 指定了网络中使用的非线性激活函数类型。
        # 定义了init_weights()函数之后需要再类初始化函数中调用。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


# 一般的项目文件中，model.py包含模型类的定义，而其中需要有模型参数的初始化函数


if __name__ == "__main__":
    net = Net()
    params = list(net.parameters())
    print(net.conv1.bias.data)
#    print(params[0].data)
    print("\n")
    print(net)
