import os

import torch
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml

import autovideo
ROOT_PATH = autovideo.__path__[0]
MODEL_PATH = os.path.join(ROOT_PATH, 'recognition/eco_utils/eco/ECO.yaml') 

class ECOModel(nn.Module):
    def __init__(self, model_path=MODEL_PATH, num_classes=101,
                       num_segments=4, pretrained_parts='both'):

        super(ECOModel, self).__init__()

        self.num_segments = num_segments

        self.pretrained_parts = pretrained_parts

        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat' and op != 'Eltwise':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True if op == 'Conv3d' else True, num_segments=num_segments)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            elif op == 'Concat':
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = self._channel_dict[in_var[0]]
                self._channel_dict[out_var[0]] = channel


    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct' and op[1] != 'Eltwise':
                # first 3d conv layer judge, the last 2d conv layer's output must be transpose from 4d to 5d
                if op[0] == 'res3a_2':
                    inception_3c_output = data_dict['inception_3c_double_3x3_1_bn']
                    inception_3c_transpose_output = torch.transpose(inception_3c_output.view((-1, self.num_segments) + inception_3c_output.size()[1:]), 1, 2)
                    data_dict[op[2]] = getattr(self, op[0])(inception_3c_transpose_output)
                else:
                    data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                    # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            elif op[1] == 'Eltwise':
                try:
                    data_dict[op[2]] = torch.add(data_dict[op[-1][0]], 1, data_dict[op[-1][1]])
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
                # x = data_dict[op[-1]]
                # data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        # print output data size in each layers
        # for k in data_dict.keys():
        #     print(k,": ",data_dict[k].size())
        # exit()

        # "self._op_list[-1][2]" represents: last layer's name(e.g. fc_action)
        return data_dict[self._op_list[-1][2]]
