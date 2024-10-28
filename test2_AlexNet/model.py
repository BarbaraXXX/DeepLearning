# AlexNet模型结构
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weight=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class AlexNet1(nn.Module):
    def __init__(self, num_classes=1000, init_weight=False):
        # 输出分类数可以自行设置， 是否初始化也可以自行设置
        super(AlexNet1, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 256, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        # 原始网络输出种类为1000， 现分类为5分类，输出通道数改为5
#        self.fc4 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(2048, num_classes)
        if init_weight:
            self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        
#        x = self.fc4(x)
#        x = F.relu(x)
#        x = F.dropout(x, p=0.5)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
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
    alexnet = AlexNet(num_classes=10, init_weight=True)
    print(alexnet)
