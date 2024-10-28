import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model_rectified import GoogLeNet


def main():
    # 超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device} to training the model')

    batch_size = 128
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    num_workers = 4
    print(f'the batch size is: {batch_size}')
    print(f'using {num_workers} workers to load data')

    epochs = 20
    lr = 1e-4
    save_path = '/root/test4_GoogLeNet/save_model'
    log_path = '/root/test4_GoogLeNet/logs'
    data_root = '/root/dataset'
    image_path = os.path.join(data_root, 'mini_imagenet')

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
#    data_transform = {
#        "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#        "test": transforms.Compose([transforms.Resize((224, 224)),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        
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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, num_workers=num_workers)

    print(f'using {train_num} images to train, {test_num} images to test')

    net = GoogLeNet(num_classes=100, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0.0
    train_steps = len(train_loader)
    # 中断
    history_acc = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc, test_acc = 0.0, 0.0
        # 更新前模型第0层权重
        before = list(net.parameters())[0].clone()
        for step, data in enumerate(train_bar):
            # train_acc = 0.0
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            loss.backward()
            optimizer.step()
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            # optimizer.step()


            running_loss += loss.item()

            train_bar.desc = 'train epoch [{} / {}] loss:{:.3f}'.format(epoch + 1,
                                                                        epochs, loss)

        train_accuracy = train_acc / train_num
        print(f'the{epoch + 1} epoch train loss is: {running_loss:.3f}')
        print(f'the{epoch + 1} epoch train acc is: {train_accuracy:.3f}')
        
        # 检验模型训练情况
        after = list(net.parameters())[0].clone()
        print(f'模型第0层更新幅度：{torch.sum(after - before)}')
        
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
        print(f'the {epoch + 1} epoch test acc is:{test_accuracy:.3f}')
        if test_accuracy > best_acc:
            best_acc = test_accuracy
        total_loss = running_loss / len(train_loader)
        with SummaryWriter(log_path, comment='GoogLeNet') as writer:
            writer.add_scalars('GoogLeNet', {'loss': total_loss, 'train_acc': train_accuracy,
                                             'test_acc': test_accuracy}, epoch + 1)
        
        scheduler.step()
        # history_acc[epoch] = test_accuracy
        history_acc.append(train_accuracy)
        if epoch > 10 and history_acc[epoch] == history_acc[epoch - 1] and history_acc[epoch] == history_acc[epoch - 2]:
            break

        print(f'the {epoch + 1} epoch finished')

    print(f'the training finished')


if __name__ == '__main__':
    main()
