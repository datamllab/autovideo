'''
The code has been derived from https://github.com/jfzhang95/pytorch-video-recognition

MIT License

Copyright (c) 2018 Pyjcsx
Copyright (c) 2021 DATA Lab at Texas A&M University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import typing
import uuid
from urllib.parse import urlparse
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from torch.autograd import Variable
from torch.nn.init import normal, constant
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase
from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions, get_frame_list, get_video_loader, adjust_learning_rate, logger

pretrained_url = "https://drive.google.com/u/1/uc?export=download&confirm=Z7Yt&id=19NWziHWh1LgCcHU34geoKwYezAogv9fX"
pretrained_path = "weights/c3d-pretrained.pth"
__all__ = ('C3DPrimitive',)
Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(SupervisedParamsBase):
    pass

class Hyperparams(SupervisedHyperparamsBase):
    num_workers = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        default=2,#4
        description='The number of subprocesses to use for data loading. 0 means that the data will be loaded in the '
                    'main process.'
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The batch size of training"
    )
    epochs = hyperparams.Hyperparameter[int](
        default=50,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="How many epochs to be trained"
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.01, #1e-3
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    momentum = hyperparams.Hyperparameter[float](
        default=0.9,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The momentum of the optimizer"
    )
    weight_decay = hyperparams.Hyperparameter[float](
        default=1e-7,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    num_segments = hyperparams.Hyperparameter[int](
        default=16,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The number of segments of frames in each video per training loop"
    )
    valid_ratio = hyperparams.Hyperparameter[float](
        default=0.05,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The ratio of validation data"
    )
    modality = hyperparams.Enumeration(
        values=['RGB', 'RGBDiff', 'Flow'],
        default='RGB',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The modality of input data to be used for the model",
    )
    num_steps_per_update = hyperparams.Hyperparameter[int](
        default=4,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The num_steps to update weights"
    )



class C3DPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of C3D pre-trained on UCF101 and HMDB51 ??
    """
    metadata = construct_primitive_metadata('recognition', 'c3d')

    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        print("Loading C3D")
        self.model = C3D(51, pretrained)
        logger.info("Loaded C3D Model")
        self.model = self.model.to(self.device)

    def _fit(self, *, timeout: float = None, iterations: int = None):
        """
        Training
        """
        #Randomly split 5% data for validation
        frame_list = np.array(get_frame_list(self._media_dir, self._inputs, self._outputs))
        idx = np.array([i for i in range(len(frame_list))])
        np.random.shuffle(idx)
        train_idx, valid_idx = idx[:int(len(idx)*(1-self.hyperparams['valid_ratio']))], idx[int(len(idx)*(1-self.hyperparams['valid_ratio'])):]
        train_list, valid_list = frame_list[train_idx], frame_list[valid_idx]
        
        # Get optimizer and loss
        # optimizer = torch.optim.SGD(self.model.get_optim_policies(self.hyperparams['learning_rate']),
        #                             self.hyperparams['learning_rate'],
        #                             momentum=self.hyperparams['momentum'],
        #                             weight_decay=self.hyperparams['weight_decay'])
        # criterion = nn.CrossEntropyLoss()
        # #Scheduler divides the lr by 10 every 10 epochs
        # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        optimizer = torch.optim.Adam(self.model.parameters(),
                         lr=self.hyperparams['learning_rate'],
                         betas=(0.5, 0.999),
                         weight_decay=self.hyperparams['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()

        
        #Create Dataloaders
        train_loader = get_video_loader(video_list=train_list,
                                        crop_size=self.model.crop_size,
                                        scale_size=self.model.scale_size,
                                        input_mean=self.model.input_mean,
                                        input_std=self.model.input_std,
                                        train_augmentation=True,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=True,
                                        input_format="NCTHW")
        valid_loader = get_video_loader(video_list=valid_list,
                                        crop_size=self.model.crop_size,
                                        scale_size=self.model.scale_size,
                                        input_mean=self.model.input_mean,
                                        input_std=self.model.input_std,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=False,
                                        input_format="NCTHW")
        
        best_valid_acc = 0.0
        tmp_file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        total_steps = self.hyperparams['epochs'] * len(train_loader)
        num_steps_per_update = self.hyperparams['num_steps_per_update']
        s1, s2 = 0.125 / num_steps_per_update, 0.375 / num_steps_per_update
        lr_steps = [int(total_steps * s1), int(total_steps * s2)] #Steps after which lr decays by a factor of 10
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps)

        
        #Training Loop
        for epoch in range(self.hyperparams['epochs']):
            #Iterate over a batch of videos with num_segments in each video

            num_iter = 0
            self.model.train() 
            
            for i, (inputs,target) in enumerate(train_loader):
                num_iter += 1

                inputs, target = inputs.to(self.device), target.to(self.device)
                #inputs = Variable(inputs, requires_grad=True).to(self.device)
                #target = target.to(self.device)
                output = self.model(inputs)
                loss = criterion(output, target)
                #optimizer.zero_grad()
                loss /= num_steps_per_update
                loss.backward()
                #optimizer.step()

                if num_iter == num_steps_per_update:
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()

               
            # Evaluation
            self.model.eval()
            train_acc = compute_accuracy(train_loader, self.model, self.device)
            valid_acc = compute_accuracy(valid_loader, self.model, self.device)
            logger.info('Epoch {}, training accuracy {:5.4f}, validation accuracy {:5.4f}'.format(epoch, train_acc*100, valid_acc*100))
            #Save best model
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), tmp_file_path)

        # Load the best model with the highest accuracy on validation data
        self.model.load_state_dict(torch.load(tmp_file_path))
        self.model.eval()
        os.remove(tmp_file_path)

    def produce(self, *, inputs: container.DataFrame, timeout: float=None, iterations: int=None) -> CallResult[container.DataFrame]:
        """
        make the predictions
        """
        #Create DataLoader
        media_dir = urlparse(inputs.metadata.query_column(0)['location_base_uris'][0]).path
        test_list = get_frame_list(media_dir, inputs, test_mode=True)
        test_loader = get_video_loader(video_list=test_list,
                                        crop_size=self.model.crop_size,
                                        scale_size=self.model.scale_size,
                                        input_mean=self.model.input_mean,
                                        input_std=self.model.input_std,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=False,
                                        test_mode=True,
                                        input_format="NCTHW")

        # Make predictions
        self.model.eval()
        preds = make_predictions(test_loader, self.model, self.device)
        outputs = wrap_predictions(inputs, preds, self.__class__.metadata.query()['name'])
        return CallResult(outputs)


class C3D(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, num_classes, pretrained=False):
        self._prepare_base_model()
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(pretrained_path)
        #p_dict = load_state_dict_from_url(pretrained_url)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_optim_policies(self, lr):
        return [{'params': get_1x_lr_params(self), 'lr': lr},
                            {'params': get_10x_lr_params(self), 'lr': lr * 10}]

    def _prepare_base_model(self):
        self.crop_size = 112
        self.scale_size = 256
        self.input_mean = [0.43216, 0.394666, 0.37645]
        self.input_std = [0.22803, 0.22145, 0.216989]    

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

