'''
The code is derived from https://github.com/yysijie/st-gcn

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
import numpy as np

from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import CallResult

import torch
import torch.nn as nn

from autovideo.base.supervised_base import SupervisedParamsBase, SupervisedHyperparamsBase, SupervisedPrimitiveBase
from autovideo.utils import wrap_predictions, construct_primitive_metadata, make_predictions, get_skeleton_loader, adjust_learning_rate, logger

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
        self._inputs, self._outputs = [x[0] for x in self._inputs.values.tolist()], [x[0] for x in self._outputs.values.tolist()]
        random_choose = self.hyperparams['random_choose']
        random_move = self.hyperparams['random_move']
        window_size = self.hyperparams['window_size']


        #Randomly split validation data
        idx = np.array([i for i in range(len(self._inputs))])
        train_idx, valid_idx = idx[:int(len(idx)*(1-self.hyperparams['valid_ratio']))], idx[int(len(idx)*(1-self.hyperparams['valid_ratio'])):]
        train_inputs, train_labels = [self._inputs[x] for x in train_idx], [self._outputs[x] for x in train_idx]
        valid_inputs, valid_labels = [self._inputs[x] for x in valid_idx], [self._outputs[x] for x in valid_idx]
        #print(len(train_inputs), len(train_labels), len(valid_inputs), len(valid_labels))
        #print(train_inputs[0].shape, train_labels[0])

        # Get optimizer and loss
        optimizer = torch.optim.SGD(self.model.parameters(),
            lr=self.hyperparams['learning_rate'],
            momentum=self.hyperparams['momentum'],
            nesterov=True,
            weight_decay= self.hyperparams['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()

        # Create Dataloaders
        train_loader = get_skeleton_loader(
            inputs=train_inputs,
            labels=train_labels,
            random_choose=self.hyperparams['random_choose'],
            random_move=self.hyperparams['random_move'],
            window_size=self.hyperparams['window_size'],
            batch_size=self.hyperparams['batch_size'],
            num_workers=self.hyperparams['num_workers']
        )
        valid_loader = get_skeleton_loader(
            inputs=train_inputs,
            labels=train_labels,
            random_choose=self.hyperparams['random_choose'],
            random_move=self.hyperparams['random_move'],
            window_size=self.hyperparams['window_size'],
            batch_size=self.hyperparams['batch_size'],
            num_workers=self.hyperparams['num_workers'],
        )

        lr_steps = [20,30,40,50]

        for epoch in range(self.hyperparams['epochs']):
            self.lr, optimizer.param_groups = adjust_lr(self.hyperparams['learning_rate'], lr_steps, self.meta_info, optimizer.param_groups)
            
            self.model.train()
            for i, (inputs, target) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                #statistics
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.hyperparams['learning_rate'])
                show_iter_info(self.meta_info, 100, self.iter_info, logger)
                self.meta_info['iter'] += 1
            

            # Evaluation
            if ((epoch + 1) % self.hyperparams['eval_interval'] == 0) or (
                epoch + 1 == self.hyperparams['epochs']):

                self.model.eval()

                logger.info('Eval epoch: {}'.format(epoch))
                preds = []
                targets = []

                for i, (inputs, target) in enumerate(valid_loader):
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        pred = self.model(inputs)
                    preds.append(pred.data.cpu().numpy())
                    targets.append(target.data.cpu().numpy())
                
                preds = np.concatenate(preds)
                targets = np.concatenate(targets)
                for k in [1,5]:
                    show_topk(preds, targets, k, logger)
                
    def _init_model(self, pretrained):
        """
        Initialize the model. Loading the weights if pretrained is True
        """
        assert not pretrained, "No pretrained model available for stgcn." 
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
        _inputs = [x[0] for x in inputs.values.tolist()]
        test_loader = get_skeleton_loader(
            inputs=_inputs,
            random_choose=self.hyperparams['random_choose'],
            random_move=self.hyperparams['random_move'],
            window_size=self.hyperparams['window_size'],
            batch_size=self.hyperparams['batch_size'],
            num_workers=self.hyperparams['num_workers'],
            shuffle=False,
            test_mode=True
        )

        # Make predictions
        self.model.eval()
        preds = make_predictions(test_loader, self.model, self.device)
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
        
        self.residual_flag = residual
        self.identity = (in_channels == out_channels) and (stride == 1)
        if residual and not self.identity:
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

        if not self.residual_flag:
            res = 0
        elif self.identity:
            res = x
        else:
            res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def adjust_lr(base_lr, step, meta_info, param_groups):
    if step:
        lr = base_lr * (0.1**np.sum(meta_info['epoch']>=np.array(step)))
        for param_group in param_groups:
            param_group['lr'] = lr
        return lr, param_groups
    else:
        lr = base_lr
        return lr, param_groups

def show_iter_info(meta_info,log_interval,iter_info,logger):
    if meta_info['iter'] % log_interval == 0 :
        info_iter = '\tIter {} Done.'.format(meta_info['iter'])
        for k,v in iter_info.items():
            if isinstance(v,float):
                info_iter = info_iter + ' | {}: {:.4f}'.format(k,v)
            else:
                info_iter = info_iter + ' | {}: {}'.format(k,v)
        logger.info(info_iter) 

def show_topk(result,label,k,logger):
    rank = result.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k)* 1.0 / len(hit_top_k)
    logger.info('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
