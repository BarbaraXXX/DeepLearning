# 由于原始网络模型直接训练会发生梯度消失问题，且不知道该怎么解决，故借鉴他人经验，尝试改变模型激活函数等
# https://blog.csdn.net/NgfSIX/article/details/131577461
# 使用批量归一化Batch_sizeNormalization
# 用两个3 x 3卷积代替path3中的5 x 5卷积


import torch.nn as nn
import torch
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(Conv, self).__init__()
        # **kwargs用于接收不同路径的大卷积核的padding数以调整不同路径的输出特征图具有相同的尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, **kwargs)

        nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 在卷积之后，relu激活之前添加BN层， 批量归一化，同时relu激活采用原地修改参数， 减少内存显存的浪费

        # 在init里设定了**kwargs参数之后也需要在类定义中接收**kwargs参数
        # 否则也不会接收参数

    def forward(self, x):
        return self.relu(self.conv1(x))


class Inception(nn.Module):
    def __init__(self, in_channels, path1_oc, path2_1oc, path2_2oc, path3_1oc, path3_2oc, path4_2oc):
        super(Inception, self).__init__()
        # in_channels:输入Inception块的通道数
        # out_channels:输出Inception块的通道数
        # path1_oc:第一条路径1 x 1卷积的输出通道数，其输入通道数为Inception块的输入通道数， 即in_channels
        # path2_1oc: 第二条路径 1 x 1卷积的输出通道数， 其输入通道数为Inception块的输入通道数， 即in_channels
        # path2_2oc: 第二条路径 3 x 3卷积的输出通道数， 其输入通道数为前面 1 x 1卷积的输出通道数，即 path2_1oc
        # path3_1oc: 第三条路径 1 x 1卷积的输出通道数， 其输入通道数为Inception块的输入通道数, 即in_channels
        # path3_2oc: 第三条路径 5 x 5卷积的输出通道数， 其输入通道数为前面 1 x 1卷积的输出通道数， 即 path3_1oc

        # path4_2oc: 第四条路径 1 x 1卷积的输出通道数， 其输入通道数为前面 Max pooling 层输出通道数数，  即in_channels(池化操作不改变输出通道数)

        # out_channels: out_channels的值为四条路径最终的输出通道数之和， 即out_channels = path1_oc + path2_2oc + path3_2oc + path4_2oc

        # 在这四条路径中的卷积均为自定义的卷积操作， 即Conv类， 在进行卷积之后进行Relu激活操作
        # 在3 x 3， 5 x 5， max pooling层中还需要设置padding参数使得四条路径输出特征图尺寸相同

        self.Path1 = Conv(in_channels, path1_oc, 1, 1)
        # 第一条路径卷积核为1 x 1，不需要padding
        self.Path2 = nn.Sequential(
            Conv(in_channels, path2_1oc, 1, 1),
            Conv(path2_1oc, path2_2oc, 3, 1, padding=1))
        # 第二条路径3 x 3卷积padding=1， 卷积之后特征图尺寸不变
        self.Path3 = nn.Sequential(
            Conv(in_channels, path3_1oc, 1, 1),
            Conv(path3_1oc, path3_2oc, 3, 1, padding=1),
            Conv(path3_2oc, path3_2oc, 3, 1, padding=1))
        # Conv(path3_1oc, path3_2oc, 5, 1, padding=2))
        # 用两个3 x 3卷积代替5 x 5卷积

        # 第三条路径5 x 5卷积padding=2
        self.Path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, path4_2oc, 1, 1))
        # 第四条路径 Max Pooling padding=1
        self.out_channels = path1_oc + path2_2oc + path3_2oc + path4_2oc

    def forward(self, x):
        path1 = self.Path1(x)
        path2 = self.Path2(x)
        path3 = self.Path3(x)
        path4 = self.Path4(x)

        output = [path1, path2, path3, path4]
        return torch.cat(output, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = Conv(in_channels, 128, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    # 辅助Inception块， 用于训练时获取loss值，并加权加入到总训练loss中


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.start_path = nn.Sequential(
            Conv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.LocalResponseNorm(),论文里没给参数， 省略
            Conv(64, 64, kernel_size=1, stride=1),
            Conv(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.aux_logits = aux_logits

        self.Inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.Inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.Inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.Inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.Inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.Inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.Inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.Inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.Inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.end_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        if init_weights:
            self.init_weight()

    def forward(self, x):
        x = self.start_path(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.Inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.Inception4e(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.Inception5a(x)
        x = self.Inception5b(x)

        x = self.end_path(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    y = torch.ones(1, 3, 224, 224)
    # net = Inception(in_channels=192, path1_oc=64, path2_1oc=96, path2_2oc=128, path3_1oc=16, path3_2oc=32,
    # path4_2oc=32)
    net = GoogLeNet(num_classes=10)
    print(net(y))
