import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from model import Vgg11, Vgg13, Vgg16, Vgg16C, Vgg19

def main():
    # 超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device} to train the model')

    batch_size = 64
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'using {num_workers} dataloader workers to load data')

    epochs = 15
    lr = 1e-4
    save_path = '/root/test3_VGG/save_model'
    log_path = '/root/test3_VGG/logs'
    data_root = '/root/dataset'
    image_path = os.path.join(data_root, 'S_flower_photos')

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])}

    assert os.path.exists(image_path), '{}path does not exist.'.format(image_path)
    train_set = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                     transform=data_transform['train'])
    test_set = datasets.ImageFolder(root=os.path.join(image_path, 'test'),
                                    transform=data_transform['test'])

    train_num = len(train_set)
    test_num = len(test_set)

    flower_list = train_set.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_dir = json.dumps(cla_dict, indent=4)

    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_dir)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              num_workers=num_workers)

    print(f'using {train_num} images to train, {test_num} images to test')

    net = Vgg16(num_classes=5, init_weight=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_acc = 0.0
    train_steps = len(train_loader)
    acc_list = []
    print(f'use {lr}learning rate to learn model')
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc = 0.0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = 'train epoch [{} / {}] loss:{:.3f}'.format(epoch + 1,
                                                                        epochs, loss)
        train_accuracy = train_acc / train_num
        print(f'the {epoch + 1} epoch train loss is:{running_loss:.3f}')
        print(f'the {epoch + 1} epoch train acc is:{train_accuracy:.3f}')

        # test
        net.eval()
        test_acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                test_acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accuracy = test_acc / test_num
        print(f'the {epoch + 1} epoch test acc is:{test_accuracy}')
        if test_accuracy > best_acc:
            best_acc = test_accuracy
        total_loss = running_loss / train_steps
        with SummaryWriter(log_path, comment='VGG') as writer:
            writer.add_scalars('VGG', {'loss': total_loss, 'train_acc': train_accuracy,
                                       'test_acc': test_accuracy}, epoch + 1)
        acc_list.append(train_accuracy)
        if epoch > 3:
            if acc_list[epoch] == acc_list[epoch -1]  and acc_list[epoch] == acc_list[epoch-2]:
                print(f'the model wont converge')
                break

        print(f'the {epoch + 1} epoch finished')
    print(f'the training finished')
    print(f'the best test acc is:{best_acc}')

if __name__ == '__main__':
    main()


