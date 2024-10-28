# resnet
import torch
import torch.nn as nn
import torch.nn.functional as F


# resnet50


class BottleNeck(nn.Module):
    # 特征图尺寸的变换在3x3卷积中进行
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(4 * out_channels),
            nn.ReLU(inplace=False))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
#        x1 = self.conv1(x)
#        x1 = self.conv2(x1)
#        x1 = self.conv3(x1)

#        x1 += self.shortcut(x)
#        x1 = F.relu(x1, inplace=False)

#        return x1
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        shortcut = self.shortcut(x)
#         out += shortcut
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out, inplace=False)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=100, init_weight=False):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=False))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 在maxpooling层中会将特征图尺寸从112变为56， 第一块残差块堆叠块不需要进行特征图尺寸变小。
        self.conv2_1 = BottleNeck(64, 64, stride=1)
        self.conv2_2 = BottleNeck(256, 64, stride=1)
        self.conv2_3 = BottleNeck(256, 64, stride=1)

        self.conv3_1 = BottleNeck(256, 128, stride=2)
        self.conv3_2 = BottleNeck(512, 128, stride=1)
        self.conv3_3 = BottleNeck(512, 128, stride=1)
        self.conv3_4 = BottleNeck(512, 128, stride=1)

        self.conv4_1 = BottleNeck(512, 256, stride=2)
        self.conv4_2 = BottleNeck(1024, 256, stride=1)
        self.conv4_3 = BottleNeck(1024, 256, stride=1)
        self.conv4_4 = BottleNeck(1024, 256, stride=1)
        self.conv4_5 = BottleNeck(1024, 256, stride=1)
        self.conv4_6 = BottleNeck(1024, 256, stride=1)

        self.conv5_1 = BottleNeck(1024, 512, stride=2)
        self.conv5_2 = BottleNeck(2048, 512, stride=1)
        self.conv5_3 = BottleNeck(2048, 512, stride=1)

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 自适应平均池化， 将输入特征图平均池化为指定的尺寸，输入特征图的尺寸不会影响输出
        self.fc = nn.Linear(2048, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        if init_weight:
            self.init_weights()

    def forward(self,x):
        out = self.maxpool(self.conv1(x))

        out = self.conv2_3(self.conv2_2(self.conv2_1(out)))

        out = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(out))))

        out = self.conv4_6(self.conv4_5(self.conv4_4(self.conv4_3(self.conv4_2(self.conv4_1(out))))))

        out = self.conv5_3(self.conv5_2(self.conv5_1(out)))
        out = torch.flatten(self.Avgpool(out), 1)
        out = self.fc(out)
#         out = self.softmax(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = ResNet(num_classes=100)

    y = torch.ones(1, 3, 224, 224)
    print(net(y))

    