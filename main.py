'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
# import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import sys
import argparse

from models import * 
from utils import progress_bar,misclassified_images,plot_gradcam,resume_from_checkpoint,plot_loss_curves


class AlbumentationDataset(torchvision.datasets.CIFAR10):
    """
  PyTorch dataset class for a subset of CIFAR-10 with Albumentations transformations.

  Args:
      data_root (str): Path to the CIFAR-10 dataset root directory.
      indices (list): List of indices representing the subset of images to use.
      transform (albumentations.Compose, optional): Albumentations transformation pipeline (default: None).
  """
    # def __init__(self, root="./data/", train=True, download=True, transform=None):
        # super().__init__(root=root, train=train, download=download, transform=transform)
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def transform_data(dataset,train_ratio, batch_size, workers):
    # Define transformations; split the dataset into train and test; return the dataset loaders
    print('==> Preparing data..')
    transform_train = A.Compose([
        A.RandomCrop(32, 32, padding=4),
        A.HorizontalFlip(),
        A.CoarseDropout(num_holes_range=(1, 1), max_height=16, max_width=16, fill_value=0.4914*255),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])

    transform_test = A.Compose([
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])
    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.targets, test_size=(1-train_ratio), random_state=42)

    trainset = AlbumentationDataset(train_data, train_labels, transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=int(workers))

    testset = AlbumentationDataset(test_data, test_labels, transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=int(workers))
    return trainloader, testloader


# Training
def train(net, trainloader, optimizer, criterion, device, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / len(trainloader)

def test(net, testloader, criterion, device, epoch, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        script_dir = os.path.dirname(__file__)  # Get the script's directory
        target_dir = os.path.join(script_dir, 'checkpoint')  # Join the path with the directory name

        if not os.path.exists(target_dir):
            os.makedirs('checkpoint')
        torch.save(state, os.path.join(target_dir,'ckpt.pth'))
        best_acc = acc
    return test_loss / len(testloader)


def main(args=None):
        
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='optimizer to use')
    parser.add_argument('--scheduler', action='store_true', help='use learning rate scheduler')
    parser.add_argument('--model', default='ResNet18', choices=['ResNet18', 'ResNet34'], help='model to use')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='ratio of train dataset')
    parser.add_argument('--workers', default=2, type=float, help='number of workers for dataloader')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    trainloader, testloader = transform_data(dataset,train_ratio=args.train_ratio, batch_size=args.batch_size, workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    if args.model=='ResNet18':
        net = ResNet18()
    else:
        net = ResNet34()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        best_acc, start_epoch = resume_from_checkpoint(net)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Train and Test
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(net, trainloader, optimizer, criterion, device, epoch)
        test_loss = test(net, testloader, criterion, device, epoch, best_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if args.scheduler:
            scheduler.step()
    
    # Plot loss curves
    plot_loss_curves(train_losses,test_losses,plot_file='LossCurve.png')

    # Misclassified Images
    misclassified = misclassified_images(net, testloader, classes, device, num_images=10, plot_file="misclassified.png")
    
    # GradCAM
    plot_gradcam(misclassified, net, device, classes, plot_file='gradcam.png')

if __name__ == "__main__":
    main(args=None)