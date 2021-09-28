'''
The code is derived from https://github.com/yysijie/st-gcn
'''

from logging import Logger
import os
import math
from d3m.container.list import L
import numpy as np
import uuid
from urllib.parse import urlparse

from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult

import torch
import torch.nn as nn
from torch.nn.init import normal, constant
from torch.hub import load_state_dict_from_url
from torch.autograd import Variable
import torchvision

import sys
import pickle
from torchvision import datasets, transforms

from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase
#from autovideo.utils.transforms import *
from autovideo.utils import wrap_predictions, construct_primitive_metadata, compute_accuracy, make_predictions_stgcn, get_frame_list, get_video_loader, adjust_learning_rate, logger
from autovideo.utils.stgcn_utils import Feeder, show_iter_info, show_epoch_info, train_info, test_info, show_topk

import autovideo.utils.stgcn_utils
from autovideo.utils.stgcn_utils import get_skeleton_list, get_skeleton_train_valid,random_choose, random_move, auto_pading, weights_init, get_hop_distance, normalize_digraph, normalize_undigraph, downsample, temporal_slice, mean_subtractor, random_shift, openpose_match, top_k_by_category, adjust_lr
#Is this mandatory?
#pretrained_url = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/tsn2d_kinetics400_rgb_r50_seg3_f1s1-b702e12f.pth"

__all__ = ('STGCNPrimitive',)
Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(SupervisedParamsBase):
    pass

class Hyperparams(SupervisedHyperparamsBase):
    num_workers = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        default=0,
        description='The number of subprocesses to use for data loading. 0 means that the data will be loaded in the '
                    'main process.'
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=64, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The batch size of training"
    )
    test_batch_size = hyperparams.Hyperparameter[int](
        default=64, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The batch size of testing"
    )
    epochs = hyperparams.Hyperparameter[int](
        default=50,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="How many epochs to be trained"
    )
    window_size = hyperparams.Hyperparameter[int](
        default=150,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Window Size for training"
    )
    random_choose = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Random Choose for Training"
    )
    random_move = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Random Move for Training"
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.1, #base_lr from train.yaml
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
    )
    momentum = hyperparams.Hyperparameter[float](
        default=0.9,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The momentum of the optimizer"
    )
    weight_decay = hyperparams.Hyperparameter[float](
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The learning rate of the optimizer"
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
    save_interval = hyperparams.Hyperparameter[int](
        default=10,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="the interval for storing models (#iteration)'"
    )
    log_interval = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="the interval for printing messages (#iteration)'"
    )
    eval_interval = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="the interval for evaluating models (#iteration)'"
    )
    step = hyperparams.Enumeration(
        values=[20, 30, 40, 50],
        default= 20,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Step to choose for optimization",
    )
    modality = hyperparams.Enumeration(
        values=['RGB', 'RGBDiff', 'Flow'],
        default='RGB',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The modality of input data to be used for the model",
    )

class STGCNPrimitive(SupervisedPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Implementation of STGCN
    """
    metadata = construct_primitive_metadata('recognition', 'stgcn')
 
    def get_params(self) -> Params:
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        super().set_params(params=params)

    def _fit(self, *, timeout: float = None, iterations: int = None):
        """
        Training
        """
        #Split train and Validation data
        train_data, train_label, val_data, val_label = get_skeleton_train_valid(self._media_dir, self._inputs, self.hyperparams['valid_ratio'], self._outputs)
        random_choose = self.hyperparams['random_choose']
        random_move = self.hyperparams['random_move']
        window_size = self.hyperparams['window_size']

        
        train_data_loader = torch.utils.data.DataLoader(
            dataset = Feeder(train_data, train_label, random_choose, random_move, window_size),
            batch_size = self.hyperparams['batch_size'],
            shuffle = True,
            num_workers = self.hyperparams['num_workers'],
            drop_last = True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset = Feeder(val_data, val_label),
            batch_size = self.hyperparams['test_batch_size'],
            shuffle = False,
            num_workers = self.hyperparams['num_workers']
        )
        
        optimizer = torch.optim.SGD(self.model.parameters(),
        lr=self.hyperparams['learning_rate'],
        momentum=self.hyperparams['momentum'],
        nesterov=True,
        weight_decay= self.hyperparams['weight_decay']
        )
        lr_steps = [20,30,40,50]
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.hyperparams['epochs']):
            self.meta_info['epoch'] = epoch

            logger.info('Training epoch: {}'.format(epoch))
            
            self.model.train()
            self.lr, optimizer.param_groups = adjust_lr(self.hyperparams['learning_rate'], lr_steps, self.meta_info, optimizer.param_groups)

            loss_value = []
            loader = train_data_loader
            for data, label in loader:
                #get data
                data = data.float().to(self.device)
                label = torch.Tensor(label.float())
                label = label.long().to(self.device)

                #forward
                output = self.model(data)
                loss = criterion(output,label)
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #statistics
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.hyperparams['learning_rate'])
                loss_value.append(self.iter_info['loss'])
                log_interval = 100
                show_iter_info(self.meta_info,log_interval,self.iter_info,logger)
                self.meta_info['iter'] += 1

            self.epoch_info['mean_loss'] = np.mean(loss_value)
            show_epoch_info(self.epoch_info,logger)
            logger.info('Done.')

            #evaluation
            if ((epoch + 1) % self.hyperparams['eval_interval'] == 0) or (
                epoch + 1 == self.hyperparams['epochs']):
                logger.info('Evaluation Start:')

                self.model.eval()

                logger.info('Eval epoch: {}'.format(epoch))
                test_loss_value = []
                test_result_frag = []
                test_label_frag = []
                test_loader = test_data_loader

                for test_data, test_label in test_loader:
                    #get data
                    test_data = test_data.float().to(self.device)
                    test_label = test_label.long().to(self.device)
                    #inference
                    with torch.no_grad():
                        test_output = self.model(test_data)
                    test_result_frag.append(test_output.data.cpu().numpy())
                    #get loss
                    test_loss = criterion(test_output,test_label)
                    test_loss_value.append(test_loss.item())
                    test_label_frag.append(test_label.data.cpu().numpy())
                
                self.result = np.concatenate(test_result_frag)
                self.label = np.concatenate(test_label_frag)
                self.epoch_info['mean_loss'] = np.mean(test_loss_value)
                show_epoch_info(self.epoch_info,logger)
                top_k = [1,5]
                for k in top_k:
                    show_topk(self.result, self.label, k, logger)

                #test_info(self.iter_info, self.meta_info, self.epoch_info, log_interval, logger)
                logger.info('Done.\n')

                
    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        #Load STGCN model
        graph_args_dict = {'layout':'openpose','strategy':'spatial'}
        self.model = Model(in_channels=3,num_class=400,graph_args=(graph_args_dict),edge_importance_weighting=True)
        self.model.apply(weights_init)

        self.iter_info =dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.model = self.model.to(self.device)
        
    def produce(self, *, inputs: container.DataFrame, timeout: float=None, iterations: int=None) -> CallResult[container.DataFrame]:
        """
        make the predictions
        """
        media_dir = urlparse(inputs.metadata.query_column(0)['location_base_uris'][0]).path
        test_data = get_skeleton_list(media_dir, inputs, test_mode=True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset = ValidFeeder(test_data),
            batch_size = self.hyperparams['batch_size'],
            shuffle = False,
            num_workers = self.hyperparams['num_workers'])

        self.model.eval()
        preds = make_predictions_stgcn(test_data_loader, self.model, self.device)
        outputs = wrap_predictions(inputs, preds, self.__class__.metadata.query()['name'])

        return CallResult(outputs)

        


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


#GRAPH
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

#T-GCN
class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

#st_gcn convolution
class st_gcn(nn.Module):

    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 label,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        #self.data = data
        #self.label = label
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(data,label)
    def load_data(self, data,label):
        # data: N C V T M
        self.label = label
        self.data = data
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        return data_numpy, label

class ValidFeeder(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False):
        self.debug = debug
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(data)
    def load_data(self, data):
        # data: N C V T M
        self.data = data
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        
        # processing
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        return data_numpy
