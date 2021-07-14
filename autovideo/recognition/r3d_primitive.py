'''
The code is derived from https://github.com/kenshohara/3D-ResNets-PyTorch

MIT License

Copyright (c) 2017 Kensho Hara
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
import numpy as np
import torch
import torch.nn as nn
import uuid
from urllib.parse import urlparse

from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult

from torch.nn.init import normal, constant
from torch.hub import load_state_dict_from_url
from torchvision import transforms

from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase
from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions, get_frame_list, get_video_loader, adjust_learning_rate, logger
from torch.nn.modules.utils import _triple

import math
from torch import optim
from torch.optim import SGD, lr_scheduler
pretrained_path = 'weights/r3d50_K_200ep.pth'
from .R2p1D.opts import parse_opts
from .R2p1D.model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from .R2p1D.mean import get_mean_std
from .R2p1D.utils import get_lr


__all__ = ('R3DPrimitive',)
Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(SupervisedParamsBase):
    pass

class Hyperparams(SupervisedHyperparamsBase):
    num_workers = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        default=4,
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
        default=0.001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    momentum = hyperparams.Hyperparameter[float](
        default=0.9,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The momentum of the optimizer"
    )
    weight_decay = hyperparams.Hyperparameter[float](
        default=5e-4,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Weight Decay"
    )
    num_segments = hyperparams.Hyperparameter[int](
        default=3,
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


class R3DPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of R3D
    """

    metadata = construct_primitive_metadata('recognition', 'r3d')

    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        
        self.opt = self._get_opt()
        self.opt.device = self.device
        self.opt.n_finetune_classes = len(np.unique(self._outputs.values))

        if pretrained:
            self.model = generate_model(self.opt)
            self.model = load_pretrained_model(self.model, self.opt.pretrain_path, self.opt.model,self.opt.n_finetune_classes)
        else:
            self.model = generate_model(self.opt)

        #For pre-trained model modify the last layer to output 51 for HMDB, 101 for UCF and so on.
        #num_classes = len(np.unique(self._outputs.values))
        #self.model.fc = nn.Linear(2048, num_classes)
        
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
        parameters = self.model.parameters()
        finetune = False
        if finetune == True:
            self.opt.ft_begin_module = 'fc'
            print("FINE TUNING ONLY")
            parameters = get_fine_tuning_parameters(self.model, self.opt.ft_begin_module)
        optimizer = torch.optim.SGD(parameters,
                                    self.hyperparams['learning_rate'],
                                    momentum=self.hyperparams['momentum'],
                                    weight_decay=self.hyperparams['weight_decay'],
                                    nesterov=self.opt.nesterov)
                                    
        if self.opt.lr_scheduler == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=self.opt.plateau_patience)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 self.opt.multistep_milestones)
        criterion = nn.CrossEntropyLoss()
        #Create Dataloaders
        train_loader = get_video_loader(video_list=train_list,
                                        crop_size=self.opt.sample_size,
                                        scale_size=112,
                                        input_mean=self.opt.mean,
                                        input_std=self.opt.std,
                                        train_augmentation=True,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=True,
                                        input_format="NCTHW")
        valid_loader = get_video_loader(video_list=valid_list,
                                        crop_size=self.opt.sample_size,
                                        scale_size=112,
                                        input_mean=self.opt.mean,
                                        input_std=self.opt.std,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=False,
                                        input_format="NCTHW")
        best_valid_acc = 0.0
        tmp_file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))
        
        #Training Loop
        for epoch in range(self.hyperparams['epochs']):
            current_lr = get_lr(optimizer)
            #print(epoch, "EPOCH")
            #Iterate over a batch of videos with num_segments in each video
            self.model.train()
            for i, (inputs,target) in enumerate(train_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.opt.lr_scheduler == 'multistep':
                scheduler.step()
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
        #mean,std = get_mean_std(self.opt.value_scale, dataset=self.opt.mean_dataset)
        #Create DataLoader
        media_dir = urlparse(inputs.metadata.query_column(0)['location_base_uris'][0]).path
        test_list = get_frame_list(media_dir, inputs, test_mode=True)
        test_loader = get_video_loader(video_list=test_list,
                                        crop_size=112,
                                        scale_size=112,
                                        input_mean=[0.4345, 0.4051, 0.3775],
                                        input_std=[0.2768, 0.2713, 0.2737],
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




    
    def _get_opt(self):
        opt = parse_opts()
        opt.dataset='hmdb51'
        opt.n_pretrain_classes = 700
        #opt.n_classes = 51
        opt.resnet_shortcut = 'B'
        self.model_name = 'r3d'
        if self.model_name=='r3d':
            print("Loaded R3D")
            opt.pretrain_path = pretrained_path
            opt.model = 'resnet'
            opt.model_depth = 50
            if opt.pretrain_path is not None:
                opt.n_finetune_classes = opt.n_classes
                opt.n_classes = opt.n_pretrain_classes
                
        opt.multistep_milestones = [20, 40, 60] #LR decay ;default for kinetics?: [50, 100, 150]
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
        opt.begin_epoch = 1
        opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
        opt.n_input_channels = 3               
        return opt


































class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        if first_conv:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding

            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)

            # from the official code, first conv's intermed_channels = 45
            intermed_channels = 45

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size =  (1, kernel_size[1], kernel_size[2])
            spatial_stride =  (1, stride[1], stride[2])
            spatial_padding =  (0, padding[1], padding[2])

            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)

            # compute the number of intermediary channels (M) using formula
            # from the paper section 3.5
            intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                                (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                        stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)

            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                        stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()



    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), first_conv=True)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class R2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R2Plus1DClassifier, self).__init__()

        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res2plus1d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res2plus1d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

