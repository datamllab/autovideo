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

import logging
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

from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions, \
    get_frame_list, get_video_loader, adjust_learning_rate, logger


from .eco_primitive import ECO

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

__all__ = ('ECOfullPrimitive')
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


class ECOFullPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of ECOfull
    """
    metadata = construct_primitive_metadata('recognition', 'eco_full')

    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

    def _fit(self, *, timeout: float = None, iterations: int = None):
        """
        Training
        """
        # Randomly split 5% data for validation
        #root = logging.getLogger()
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
        #root.error('ECO-full')
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
            print('computing train_acc:')
            train_acc = compute_accuracy(train_loader, self.model, self.device)
            print('computing valid_acc:')
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
        #print('fit', self.model.new_fc.weight)
        os.remove(tmp_file_path)

    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        # Load ECO Full model

        if pretrained:
            self.model = ECO(400, self.hyperparams['num_segments'], 'both', self.hyperparams['modality'],
                             base_model='ECOfull', consensus_type='identity', dropout=0.8, partial_bn=False)

        else:
            self.model = ECO(400, self.hyperparams['num_segments'], 'scratch', self.hyperparams['modality'],
                             base_model='ECOfull', consensus_type='identity', dropout=0.8, partial_bn=False)


        model_dict = self.model.state_dict()
        new_state_dict = init_ECOfull(model_dict, self.model.pretrained_parts)

        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        
        print("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    #print("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    #print("{} init as: kaiming normal".format(k))
                    kaiming_normal_(new_state_dict[k])
            elif 'bias' in k:
                #print("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)

        print("------------------------------------")

        num_classes = len(np.unique(self._outputs.values))
        self.model.load_state_dict(new_state_dict)

        self.model.new_fc = nn.Linear(1536, num_classes)



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


def init_ECOfull(model_dict, pretrained_parts):

    if pretrained_parts == "scratch":

        new_state_dict = {}

    elif pretrained_parts == "finetune":
        pretrained_dict = torch.load(pretrained_path_finetune)
        print(("=> loading model-finetune-url: '{}'".format(pretrained_path_finetune)))

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
