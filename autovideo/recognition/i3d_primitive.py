'''
This code has been derived from https://github.com/hassony2/kinetics_i3d_pytorch

MIT License

Copyright (c) 2017 Yana Hasson
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


import math

import os
import numpy as np
import uuid
from urllib.parse import urlparse

from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult

import torch
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase
from autovideo.utils.transforms import *
from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions, get_frame_list, get_video_loader, adjust_learning_rate, logger

# The pretrained model is from https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth
pretrained_url = "https://i3d.s3.us-east-2.amazonaws.com/model_rgb-9729d6f0.pth"

__all__ = ('I3DPrimitive',)
Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(SupervisedParamsBase):
    pass

class Hyperparams(SupervisedHyperparamsBase):
    num_workers = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        default=2,
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
        default=0.01,
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
        values=['RGB', 'Flow'],
        default='RGB',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The modality of input data to be used for the model",
    )
    down_rate = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The down rate"
    )
    dropout_prob = hyperparams.Hyperparameter[float](
        default=0.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The dropout probability"
    )
    num_steps_per_update = hyperparams.Hyperparameter[int](
        default=4,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The num_steps to update weights"
    )

class I3DPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of I3D
    """
    metadata = construct_primitive_metadata('recognition', 'i3d')
 
    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

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
        #root.error('Using Adam optimizer')
        optimizer = torch.optim.Adam(self.model.parameters(),
                         lr=self.hyperparams['learning_rate'],
                         betas=(0.5, 0.999),
                         weight_decay=self.hyperparams['weight_decay'])
        # root.error('Using SGD optimizer')
        # optimizer = torch.optim.SGD(self.model.parameters(),
        #                             self.hyperparams['learning_rate'],
        #                             momentum=self.hyperparams['momentum'],
        #                             weight_decay=self.hyperparams['weight_decay'])
        # criterion = F.binary_cross_entropy_with_logits
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

        #root.error('hyperparams {}'.format(self.hyperparams))
        best_valid_acc = 0.0
        total_steps = self.hyperparams['epochs'] * len(train_loader)
        num_steps_per_update = self.hyperparams['num_steps_per_update']
        s1, s2 = 0.125 / num_steps_per_update, 0.375 / num_steps_per_update
        lr_steps = [int(total_steps * s1), int(total_steps * s2)] #Steps after which lr decays by a factor of 10
        #root.error('lr_steps: {}, total_steps: {}, num_steps_per_update: {}'.format(lr_steps, total_steps,num_steps_per_update))
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps)
        tmp_file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))
        #root.error('tmp: {}'.format(tmp_file_path))

        #Training Loop
        for epoch in range(self.hyperparams['epochs']):
            self.model.train()
            num_iter = 0

            #Iterate over a batch of videos with num_segments in each video
            for i, (inputs,target) in enumerate(train_loader):
                num_iter += 1
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                # labels = self._augment_targets(target)

                per_frame_logits = self.model(inputs)
                # upsample to input size
                # per_frame_logits = F.upsample(per_frame_logits, self.hyperparams['num_segments'], mode='linear')

                # compute localization loss
                # loc_loss = criterion(per_frame_logits, labels)

                # cosshmpute classification loss
                # cls_loss = criterion(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])

                # loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                loss = criterion(per_frame_logits, target)
                loss /= num_steps_per_update
                loss.backward()

                #if i % 50 == 0:
                #    root.error('train {:03d} loss {:5.4f} lr {:5.6f}'.format(i, loss.item(), lr_sched.get_lr()[0]))

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
     
    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        #Load I3D model
        self.model = I3D(num_classes=400, modality=self.hyperparams['modality'].lower(),
                         dropout_prob=self.hyperparams['dropout_prob'])
        root = logging.getLogger()
        if pretrained:
            state_dict = load_state_dict_from_url(pretrained_url)
            self.model.load_state_dict(state_dict)
            freeze = True
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                root.error('Use pretrained model (freeze) {}'.format(pretrained_url))
            else:
                root.error('Use pretrained model (no freeze) {}'.format(pretrained_url))
        else:
            root.error('Do not use pretrained model')
        #For pre-trained model modify the last layer to output 51 for HMDB, 101 for UCF and so on.
        num_classes = len(np.unique(self._outputs.values))
        self.num_classes = num_classes
        self.model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                            out_channels=num_classes,
                                            kernel_size=(1, 1, 1),
                                            padding='VALID',
                                            activation=None,
                                            use_bias=True,
                                            use_bn=False)
        self.model = self.model.to(self.device)
        
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

    def _augment_targets(self, targets):
        num_segments = self.hyperparams['num_segments']
        num_classes = self.num_classes
        n = targets.size(0)

        target_expansion = targets.view(-1, 1).expand(-1, num_segments).reshape(-1, 1)
        target_onehot = torch.zeros((n * num_segments, num_classes), dtype=torch.float32)
        target_onehot.scatter_(1, target_expansion, 1)
        target_onehot = target_onehot.reshape(n, num_segments, num_classes).permute(0, 2, 1).to(self.device)
        return target_onehot

def get_padding_shape(filter_shape, stride, mod=0):
    """Fetch a tuple describing the input padding shape.

    NOTES: To replicate "TF SAME" style padding, the padding shape needs to be
    determined at runtime to handle cases when the input dimension is not divisible
    by the stride.
    See https://stackoverflow.com/a/49842071 for explanation of TF SAME padding logic
    """
    def _pad_top_bottom(filter_dim, stride_val, mod):
        if mod:
            pad_along = max(filter_dim - mod, 0)
        else:
            pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for idx, (filter_dim, stride_val) in enumerate(zip(filter_shape, stride)):
        depth_mod = (idx == 0) and mod
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val, depth_mod)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)

    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.stride = stride
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            # This is not strictly the correct map between epsilons in keras and
            # pytorch (which have slightly different definitions of the batch norm
            # forward pass), but it seems to be good enough. The PyTorch formula
            # is described here:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
            tf_style_eps = 1E-3
            self.batch3d = torch.nn.BatchNorm3d(out_channels, eps=tf_style_eps)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            # Determine the padding to be applied by examining the input shape
            pad_idx = inp.shape[2] % self.stride[0]
            pad_op = self.pads[pad_idx]
            inp = pad_op(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.stride = stride
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
            self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        pad_idx = inp.shape[2] % self.stride[0]
        pad_op = self.pads[pad_idx]
        inp = pad_op(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)
        self._prepare_base_model(name)

    def forward(self, inp):
        # Preprocessing
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        # if not self.training:
        #     out = self.softmax(out)
        return out

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
        elif base_model == 'BNInception':
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'flow':
                self.input_mean = [128]
        elif 'inception' in base_model:
            self.input_size = 224
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))

