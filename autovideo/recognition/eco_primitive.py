'''
https://github.com/mzolfaghari/ECO-pytorch

BSD 2-Clause License

Copyright (c) 2017, Multimedia Laboratary, The Chinese University of Hong Kong
All rights reserved.
Copyright (c) 2021 DATA Lab at Texas A&M University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import os
import requests
import math
import numpy as np
import uuid
from urllib.parse import urlparse

from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult

import torch
from torch import nn
from torch.nn.init import normal, constant, constant_, kaiming_normal_, xavier_uniform_
from torch.hub import load_state_dict_from_url
import torchvision

from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase

from autovideo.utils.transforms import *
from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions, \
    get_frame_list, get_video_loader, adjust_learning_rate, logger

import autovideo
# ROOT_PATH = autovideo.__path__[0]
# pretrained_file_id = "17SnoxH8tkuUCvW-4ifa4Hk7ITm93nMqI"
# destination = os.path.join(ROOT_PATH, 'model_weights', 'eco_lite_rgb_16F_kinetics_v3.pth.tar')
pretrained_url_finetune= "https://drive.google.com/file/d/11cVEZHPaNv6Bl5eTkvBA-JvE0arpUEIe/view?usp=sharing"
pretrained_url_both_2d = "https://drive.google.com/file/d/1ITB2Q8IBPI9VfqBD6xdqs0fXC3N-bKB7/view?usp=sharing"
pretrained_url_both_3d = "https://drive.google.com/file/d/1XXwuEpn1t-AbsIg5xDXBPOqS37QY0Y9X/view?usp=sharing"

pretrained_path_finetune = "weights/eco_lite_rgb_16F_kinetics_v3.pth.tar"
pretrained_path_2d = "weights/bninception_rgb_kinetics_init-d4ee618d3399.pth"
pretrained_path_3d = "weights/C3DResNet18_rgb_16F_kinetics_v1.pth.tar"

__all__ = ('ECOPrimitive')
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
        default=8,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The batch size of training"
    )
    epochs = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="How many epochs to be trained"
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    # momentum = hyperparams.Hyperparameter[float](
    #     default=0.9,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #     description="The momentum of the optimizer"
    # )
    weight_decay = hyperparams.Hyperparameter[float](
        default=5e-4,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    num_segments = hyperparams.Hyperparameter[int](
        default=8,
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


class ECOPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of ECO
    """
    metadata = construct_primitive_metadata('recognition', 'eco')

    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

    def _fit(self, *, timeout: float = None, iterations: int = None):
        """
        Training
        """
        #root = logging.getLogger()
        # Randomly split 5% data for validation
        frame_list = np.array(get_frame_list(self._media_dir, self._inputs, self._outputs))
        idx = np.array([i for i in range(len(frame_list))])
        np.random.shuffle(idx)
        train_idx, valid_idx = idx[:int(len(idx) * (1 - self.hyperparams['valid_ratio']))], idx[int(
            len(idx) * (1 - self.hyperparams['valid_ratio'])):]
        train_list, valid_list = frame_list[train_idx], frame_list[valid_idx]

        # Get optimizer and loss
        # optimizer = torch.optim.SGD(self.model.get_optim_policies(),
        #                              self.hyperparams['learning_rate'],
        #                              momentum=self.hyperparams['momentum'],
        #                              weight_decay=self.hyperparams['weight_decay'])
        optimizer = torch.optim.Adam(self.model.get_optim_policies(),
                                    self.hyperparams['learning_rate'],
                                    # momentum=self.hyperparams['momentum'],
                                    weight_decay=self.hyperparams['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        # Create Dataloaders
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
                                        shuffle=True)
        valid_loader = get_video_loader(video_list=valid_list,
                                        crop_size=self.model.crop_size,
                                        scale_size=self.model.scale_size,
                                        input_mean=self.model.input_mean,
                                        input_std=self.model.input_std,
                                        modality=self.hyperparams['modality'],
                                        num_segments=self.hyperparams['num_segments'],
                                        batch_size=self.hyperparams['batch_size'],
                                        num_workers=self.hyperparams['num_workers'],
                                        shuffle=False)

        #root.error('hyperparams {}'.format(self.hyperparams))
        #root.error('ECO-lite')
        best_valid_acc = 0.0
        lr_steps = [30, 60]  # Steps after which lr decays by a factor of 10

        tmp_file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))
        #root.error('tmp: {}'.format(tmp_file_path))
        # Training Loop
        for epoch in range(self.hyperparams['epochs']):
            adjust_learning_rate(self.hyperparams['learning_rate'],
                                 self.hyperparams['weight_decay'],
                                 optimizer,
                                 epoch,
                                 lr_steps)  # lr decay
            # Iterate over a batch of videos with num_segments in each video
            self.model.train()
            for i, (inputs, target) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                loss = criterion(output, target)
                #if i%10 ==0:
                #    root.error('train {:03d} loss {:5.4f}'.format(i, loss.item()))
                loss.backward()
                optimizer.step()

            # Evaluation
            self.model.eval()
            train_acc = compute_accuracy(train_loader, self.model, self.device)
            valid_acc = compute_accuracy(valid_loader, self.model, self.device)
            logger.info('Epoch {}, training accuracy {:5.4f}, validation accuracy {:5.4f}'.format(epoch, train_acc*100, valid_acc*100))
            #root.error(
            #    'Epoch {}: training accuracy: {:5.4f}, validation accuracy: {:5.4f}'.format(epoch, train_acc * 100, valid_acc * 100))
            # Save best model
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
        # Load ECO model

        if pretrained:
            self.model = ECO(400, self.hyperparams['num_segments'], 'both', self.hyperparams['modality'],
                             base_model='ECO', consensus_type='identity', dropout=0.8, partial_bn=False)

        else:
            self.model = ECO(400, self.hyperparams['num_segments'], 'scratch', self.hyperparams['modality'],
                             base_model='ECO', consensus_type='identity', dropout=0.8, partial_bn=False)


        model_dict = self.model.state_dict()
        new_state_dict = init_ECO(model_dict, self.model.pretrained_parts)

        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        print("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    # print("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    # print("{} init as: kaiming normal".format(k))
                    kaiming_normal_(new_state_dict[k])
            elif 'bias' in k:
                # print("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)

        print("------------------------------------")

        num_classes = len(np.unique(self._outputs.values))
        self.model.load_state_dict(new_state_dict)

        self.model.new_fc = nn.Linear(512, num_classes)



        self.model = self.model.to(self.device)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[
        container.DataFrame]:
        """
        make the predictions
        """
        # Create DataLoader
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
                                       test_mode=True)

        # Make predictions
        self.model.eval()
        preds = make_predictions(test_loader, self.model, self.device)
        outputs = wrap_predictions(inputs, preds, self.__class__.metadata.query()['name'])

        return CallResult(outputs)


def init_ECO(model_dict, pretrained_parts):

    if pretrained_parts == "scratch":

        new_state_dict = {}

    elif pretrained_parts == "finetune":
        download_file_from_google_drive(pretrained_file_id, destination)
        pretrained_dict = torch.load(destination)
        print(("=> loading model-finetune-url: '{}'".format(destination)))

        new_state_dict = {k[7:]: v for k, v in pretrained_dict['state_dict'].items() if
                          (k[7:] in model_dict) and (v.size() == model_dict[k[7:]].size())}
        print("*" * 50)
        print("Start finetuning ..")
        return new_state_dict

    elif pretrained_parts == "both":

        # Load the 2D net pretrained model
        # weight_url_2d = 'https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
        pretrained_dict_2d = torch.load(pretrained_path_2d)
        print(("=> loading model - 2D net-url:  '{}'".format(pretrained_path_2d)))

        # Load the 3D net pretrained model
        # pretrained_file_id_3d="1J2mV0Kl9pWOK0FJ23ApHnJHQ3eq76D8a"
        # destination_3d = "C3DResNet18_rgb_16F_kinetics_v1.pth.tar"
        # download_file_from_google_drive(pretrained_file_id_3d, destination_3d)
        pretrained_dict_3d = torch.load(pretrained_path_3d)
        print(("=> loading model - 3D net-url:  '{}'".format(pretrained_path_3d)))

        new_state_dict = {"base_model." + k: v for k, v in pretrained_dict_2d['state_dict'].items() if
                          "base_model." + k in model_dict}

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k[7:] in model_dict) and (v.size() == model_dict[k[7:]].size()):
                new_state_dict[k[7:]] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["base_model.res3a_2.weight"] = torch.cat(
            (res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    return new_state_dict



class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)
import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)


class ECO(nn.Module):
    def __init__(self, num_class, num_segments, pretrained_parts, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(ECO, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.pretrained_parts = pretrained_parts
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing ECO with base model: {}.
ECO Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        # zc comments
        feature_dim = self._prepare_tsn(num_class)
        # modules = list(self.modules())
        # print(modules)
        # zc comments end

        '''
        # zc: print "NN variable name"
        zc_params = self.base_model.state_dict()
        for zc_k in zc_params.items():
            print(zc_k)

        # zc: print "Specified layer's weight and bias"
        print(zc_params['conv1_7x7_s2.weight'])
        print(zc_params['conv1_7x7_s2.bias'])
        '''

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            xavier_uniform_(getattr(self.base_model, self.base_model.last_layer_name).weight)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'C3DRes18':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc8'
            self.input_size = 112
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif base_model == 'ECO':
            from .eco_utils import ECOModel
            self.base_model = ECOModel(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            
            self.base_model.last_layer_name = 'fc_final'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif base_model == 'ECOfull' :
            from .eco_utils import ECOFullModel
            self.base_model = ECOFullModel(num_segments=self.num_segments, pretrained_parts=self.pretrained_parts)
            self.base_model.last_layer_name = 'fc_final'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)


        elif base_model == 'BN2to1D':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)(num_segments=self.num_segments)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ECO, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            print("No BN layer Freezing.")

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_3d_conv_weight = []
        first_3d_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_2d_cnt = 0
        conv_3d_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                conv_2d_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_3d_cnt += 1
                if conv_3d_cnt == 1:
                    first_3d_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_3d_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_3d_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_3d_conv_weight"},
            {'params': first_3d_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_3d_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def get_optim_policies_BN2to1D(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        last_conv_weight = []
        last_conv_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                last_conv_weight.append(ps[0])
                if len(ps) == 2:
                    last_conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
             {'params': last_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "last_conv_weight"},
            {'params': last_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "last_conv_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # input.size(): [32, 9, 224, 224]
        # after view() func: [96, 3, 224, 224]
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())
        if self.base_model_name == "C3DRes18":
            before_permute = input.view((-1, sample_len) + input.size()[-2:])
            input_var = torch.transpose(before_permute.view((-1, self.num_segments) + before_permute.size()[1:]), 1, 2)
        else:
            input_var = input.view((-1, sample_len) + input.size()[-2:])
        base_out = self.base_model(input_var)

        # zc comments
        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # zc comments end
        
        if self.reshape:
          
            if self.base_model_name == 'C3DRes18':
                output = base_out
                output = self.consensus(base_out)
                return output
            elif self.base_model_name == 'ECO':
                output = base_out
                output = self.consensus(base_out)
                return output
            elif self.base_model_name == 'ECOfull':
                output = base_out
                output = self.consensus(base_out)
                return output
            else:
                # base_out.size(): [32, 3, 101], [batch_size, num_segments, num_class] respectively
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                # output.size(): [32, 1, 101]
                output = self.consensus(base_out)
                # output after squeeze(1): [32, 101], forward() returns size: [batch_size, num_class]
                return output.squeeze(1)


    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])


