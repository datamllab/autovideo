import numpy as np
import torch
import torchvision

from .dataset import VideoDataSet
from .transforms import *

from .logging_utils import logger

def compute_accuracy(loader, model, device):
    """Compute accuracy"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

def compute_accuracy_with_preds(preds, labels):
    """Compute accuracy with predictions and labels"""
    correct = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            correct += 1
    return float(correct) / len(preds)

def make_predictions(loader, model, device):
    preds = []
    confidences = []
    with torch.no_grad():
        for i, (inputs) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            softmax = torch.nn.Softmax(dim =1)
            outputs = softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)
            preds.append(predicted)
            confidences.append(outputs)
            
    preds = torch.cat(preds).cpu().detach().numpy()
    confidences = torch.cat(confidences).cpu().detach().numpy()
    logger.info("Confidence Scores: {}".format(confidences))
    return preds
    
def get_video_loader(video_list, modality, num_segments, batch_size, num_workers, crop_size=None, scale_size=None, input_mean=None, input_std=None, shuffle=False, test_mode=False, train_augmentation=False, input_format="NCHW"):
    #TODO: Modify Stack in transforms according to the architectures used. For now default value is False
    if modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
    if modality == 'RGB':
        data_length = 1
    elif modality in ['Flow', 'RGBDiff']:
        data_length = 5
    
    
    if train_augmentation == False: #shuffle false for validation and test set
        augmentation = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),])
    else:
        if modality == 'RGB':
            augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(crop_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif modality == 'Flow':
            augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(crop_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif modality == 'RGBDiff':
            augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(crop_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
    
    data_loader = torch.utils.data.DataLoader(
    VideoDataSet("",
                 video_list,
                 num_segments=num_segments, #Number of frames per video.(if >#actual frames, repeats first frame throughout)
                 new_length=data_length,
                 modality=modality,
                 test_mode=test_mode,
                 input_format=input_format,
                 image_tmpl="img_{:05d}.jpg",# if self.hyperparams['modality'] in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg"
                 transform=torchvision.transforms.Compose([
                    augmentation,
                    Stack(roll=False),
                    ToTorchFormatTensor(),
                    normalize,
                 ])),
                 batch_size=batch_size,
                 shuffle=shuffle,
                 num_workers=num_workers,
                 pin_memory=True)
    return data_loader

def adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 20 or 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']

