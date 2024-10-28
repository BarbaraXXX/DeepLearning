# train AlexNet
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import AlexNet1


def main():
    # 超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device} to train')

    batch_size = 32
    # 线程数为cpu核数， batch_size， 8这三个数中选择最小的作为线程数
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'using {num_workers}dataloader worker to load data')

    epochs = 20
    lr = 0.00002
    save_path = '/root/test2_AlexNet/save_model'
    log_path = '/root/test2_AlexNet/logs'
    data_root = '/root/dataset'
    
    writer = SummaryWriter(log_path, comment="AlexNet")
    
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])}

    image_path = os.path.join(data_root, 'S_flower_photos')
    assert os.path.exists(image_path), "{}path does not exist.".format(image_path)
    # ImageFolder 形式的数据集读取方式
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'test'),
                                        transform=data_transform['val'])

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 种类和idx的字典
    json_str = json.dumps(cla_dict, indent=4)
    # 将cla_dict字典转换为json格式，indent参数表示缩进
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        # 文件写在当前工作目录

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=4, shuffle=False,
                                              num_workers=num_workers)
    print(f'using {train_num} images to train, {test_num} images to test')

#    test_data_iter = iter(test_loader)
#    test_image, test_label = test_data_iter.next()
#    # SummaryWriter显示验证集数据
#    with SummaryWriter(log_path, comment='AlexNet') as writer:
#        writer.add_image(str(test_label), test_image, 0)
#    在使用iter获取数据时，不能用多进程读取数据

    alexnet = AlexNet1(num_classes=5, init_weight=True)

    alexnet.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(), lr=lr)

    best_acc = 0.0
    train_steps = len(train_loader)

    # train
    for epoch in range(epochs):
        alexnet.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc = 0.0
        # 用于创建进度条，在训练过程中实时显示训练进度
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = alexnet(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num
        total_loss = running_loss / train_num * 15
        print(f'the {epoch + 1} epochs‘s loss is{running_loss:.3f}')
        print(f'the {epoch + 1} epoch‘s acc is {train_accurate:.2f}')
#        with SummaryWriter(log_path, comment='AlexNet') as writer:
        # writer.add_scalar("train_loss", total_loss, epoch + 1)
        # writer.add_scalar('train_acc', train_accurate, epoch + 1)

        # test
        alexnet.eval()
        test_acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = alexnet(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                test_acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accurate = test_acc / test_num
        print(f'the {epoch + 1}epoch’s acc is{test_accurate}')
        if test_accurate > best_acc:
            best_acc = test_accurate
#        with SummaryWriter(log_path, comment='AlexNet') as writer:
        # writer.add_scalar('test_acc', test_accurate, epoch + 1)
        writer.add_scalars('acc', {'train':train_accurate, 'test':test_accurate, 'loss':total_loss}, epoch + 1)
    torch.save(alexnet.state_dict(), save_path)
    print(f'the model has saved')
    print(f'training finished')
    print(f'the best acc is{best_acc:.3f}')


if __name__ == "__main__":
    main()
