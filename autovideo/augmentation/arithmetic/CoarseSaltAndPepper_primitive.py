'''
Copyright 2021 D3M Team
Copyright (c) 2021 DATA Lab at Texas A&M University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from d3m import container
from d3m.metadata import hyperparams
import imgaug.augmenters as iaa
import typing
from autovideo.utils import construct_primitive_metadata
from autovideo.base.augmentation_base import AugmentationPrimitiveBase

__all__ = ('CoarseSaltAndPepperPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    p = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.02,0.1),
        description='Probability of changing a pixel to salt/pepper noise.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    size_px = hyperparams.Hyperparameter[typing.Union[int,tuple,None]](
        default=None,
        description=' The size of the lower resolution image from which to sample the dropout mask in absolute pixel dimensions..',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    size_percent = hyperparams.Hyperparameter[typing.Union[float,tuple,None]](
        default=0.5,
        description=' The size of the lower resolution image from which to sample the replacement mask in percent of the input image. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    per_channel = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=False,
        description='Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    min_size = hyperparams.Hyperparameter[int](
        default=3,
        description='Minimum height and width of the low resolution mask.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    


class CoarseSaltAndPepperPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Replace rectangular areas in images with white/black-ish pixel noise.
    """

    metadata = construct_primitive_metadata("augmentation", "arithmetic_CoarseSaltAndPepper")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        p = self.hyperparams["p"]
        size_px = self.hyperparams['size_px']
        size_percent = self.hyperparams["size_percent"]
        per_channel = self.hyperparams['per_channel']
        min_size = self.hyperparams['min_size']
        return iaa.CoarseSaltAndPepper(p=p,size_percent=size_percent,size_px=size_px,per_channel=per_channel,min_size=min_size,seed=seed)