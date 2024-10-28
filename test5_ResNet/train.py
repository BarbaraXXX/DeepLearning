# train resnet50
# using mini-imagenet dataset

import os.path
import sys

import torch
import torch.optim as optim
from torchvision import utils, transforms, datasets
import torch.nn as nn
import json
from tqdm import tqdm
from model import ResNet
from tensorboardX import SummaryWriter


def main():
    # 超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device} to train the model')

    batch_size = 128
    # 论文中作者使用size为128的mini batch训练模型
    num_workers = 4
    print(f'the batch_size is :{batch_size}')
    print(f'using {num_workers} workers to train the model')

    epochs = 150
    lr = 0.1
#    save_path = 'test5_ResNet/save_model'
    log_path = 'test5_ResNet/logs'
    data_root = 'dataset'
    image_path = os.path.join(data_root, 'S_flower_photos')

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])}
        
        
    assert os.path.exists(image_path), '{}path does not exist.'.format(image_path)

    train_set = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                     transform=data_transform['train'])
    test_set = datasets.ImageFolder(root=os.path.join(image_path, 'test'),
                                    transform=data_transform['test'])

    train_num = len(train_set)
    test_num = len(test_set)

    image_list = train_set.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())

    json_dir = json.dumps(cla_dict, indent=4)

    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_dir)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    print(f'using {train_num} images to train, {test_num} images to test')

    net = ResNet(num_classes=5, init_weight=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    best_acc = 0.0
    train_steps = len(train_loader)

    history_acc = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc, test_acc = 0.0, 0.0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            y_hat = net(images.to(device))
            loss = loss_function(y_hat, labels.to(device))
            loss.backward()
            predicted_y = torch.max(y_hat, dim=1)[1]
            train_acc += torch.eq(predicted_y, labels.to(device)).sum().item()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = 'train epoch [{} / {}] loss:{:.3f}'.format(epoch + 1, epochs, loss)

        train_accuracy = train_acc / train_num
        print(f'the{epoch + 1} epoch train loss is: {running_loss:.3f}')
        print(f'the{epoch + 1} epoch train acc is: {train_accuracy:.3f}')

        net.eval()
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))
                predicted_y = torch.max(outputs, dim=1)[1]
                test_acc += torch.eq(predicted_y, test_labels.to(device)).sum().item()

        test_accuracy = test_acc / test_num
        print(f'the {epoch + 1} epoch test acc is:{test_accuracy}')
        if test_accuracy > best_acc:
            best_acc = test_accuracy

        total_loss = running_loss / train_steps * 0.5
        with SummaryWriter(log_path, comment='ResNet_flower') as writer:
            writer.add_scalars('ResNet_flower', {'loss': total_loss,
                                          'train_acc': train_accuracy, 'test_acc':test_accuracy}, epoch + 1)

        history_acc.append(test_accuracy)
        if epoch > 5 and history_acc[epoch] == history_acc[epoch - 1] and history_acc[epoch] == history_acc[epoch - 2]:
            print(f'the training has been break')
            break
        print(f'the {epoch + 1} epoch finished')
    print(f'the training finished')


if __name__ == '__main__':
    main()