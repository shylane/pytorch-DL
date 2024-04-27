'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
# import torchvision.transforms as transforms
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
import argparse

from models import *
from utils import progress_bar,misclassified_images,gradcam


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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'], help='optimizer to use')
parser.add_argument('--scheduler', action='store_true', help='use learning rate scheduler')
parser.add_argument('--model', default='ResNet18', choices=['ResNet18', 'ResNet34'], help='model to use')
parser.add_argument('--train_ratio', default=0.8, type=float, help='ratio of train dataset')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def test(net, testloader, criterion, device, epoch):
    global best_acc
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
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def main(args):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
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

    # Data Split
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    train_data, test_data, train_labels, test_labels = train_test_split(dataset.data, dataset.targets, test_size=(1-args.train_ratio), random_state=42)

    trainset = AlbumentationDataset(train_data, train_labels, transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = AlbumentationDataset(test_data, test_labels, transform_test)
        # root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

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
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

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
        train(net, trainloader, optimizer, criterion, device, epoch)
        test(net, testloader, criterion, device, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if args.scheduler:
            scheduler.step()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

    # Misclassified Images
    misclassified = misclassified_images(net, testloader, device, num_images=10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, (img, target, pred) in enumerate(misclassified):
        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f'Target: {classes[target]}\nPredicted: {classes[pred]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # GradCAM
    misclassified_imgs = [img for img, target, pred in misclassified]
    targets = [target for img, target, pred in misclassified]
    preds = [pred for img, target, pred in misclassified]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, (img, target, pred) in enumerate(zip(misclassified_imgs, targets, preds)):
        heatmap = gradcam(net, img, pred, device)
        img = img.permute(1, 2, 0).cpu().numpy()
        heatmap = heatmap.cpu().numpy()
        cam = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        axes[i].imshow(superimposed_img / np.max(superimposed_img))
        axes[i].set_title(f'Target: {classes[target]}\nPredicted: {classes[pred]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(args)