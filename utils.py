'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
# import math
from typing import Optional

import torch.nn as nn
import torch.nn.init as init
# import torchvision.transforms as transforms
import numpy as np
import torch
# import torchvision.models as models
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import matplotlib.pyplot as plt


def misclassified_images(net, testloader, classes, device='cuda', num_images=10, plot_file: Optional[str]="misclassified.png"):
    net.eval()
    misclassified_images = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) #outputs.max(1)
            incorrect = (predicted != labels).nonzero().squeeze()
            for i in incorrect:
                if len(misclassified_images) < num_images:
                    misclassified_images.append((images[i], labels[i], predicted[i]))
    
    if plot_file is not None:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for i, (img, target, pred) in enumerate(misclassified_images):
            image = img.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC format
            image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
            axes[i].imshow(image)
            axes[i].set_title(f'Target: {classes[target]}\nPredicted: {classes[pred]}\nPixel size: {image.shape[0]}x{image.shape[1]}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_file)

    return misclassified_images

def gradcam(model, input_image, target_class, device):
    target_layers = [model.module.layer3[-1]]
    input_tensor = input_image.unsqueeze(0).cpu()
    image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize the image to [0, 1]
    
    # targets = target_class.unsqueeze(0).cpu()
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(target_class)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.float32(image.permute(1, 2, 0).detach().cpu().numpy()), grayscale_cam, use_rgb=True)

    # You can also get the model outputs without having to re-inference
    model_outputs = cam.outputs

    # cam = np.uint8(255*grayscale_cam)
    # cam = cv2.merge([cam, cam, cam])
    # images = np.hstack((np.uint8(255*img), cam , cam_image))
    # Image.fromarray(images)

    return visualization, grayscale_cam, cam.outputs

def plot_gradcam(misclassified, net, device, classes, plot_file='gradcam.png'):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, (img, target, pred) in enumerate(misclassified):
        viz, gray, cout = gradcam(net, img, pred, device)
        # img = img.permute(1, 2, 0).cpu().numpy()
        # heatmap = heatmap.cpu().numpy()
        # cam = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # superimposed_img = heatmap * 0.4 + img
        axes[i].imshow(viz)#(superimposed_img / np.max(superimposed_img))
        axes[i].set_title(f'Target: {classes[target]}\nPredicted: {classes[pred]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(plot_file)

def resume_from_checkpoint(net):
    #Load Checkpoint and return the best accuracy and epoch to start with to resume training
    print('==> Resuming from checkpoint..')
    script_dir = os.path.dirname(__file__)  # Get the script's directory
    target_dir = os.path.join(script_dir, 'checkpoint')  # Join the path with the directory name
    assert os.path.exists(target_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(target_dir,'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return best_acc, start_epoch

def plot_loss_curves(train_losses,test_losses,plot_file='LossCurve.png'):
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()
    plt.savefig(plot_file)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # # Go back to the center of the bar.
    # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
