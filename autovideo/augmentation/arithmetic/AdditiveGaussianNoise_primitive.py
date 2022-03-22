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

__all__ = ('AdditiveGaussianNoisePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    loc = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=0,
        description='Mean of the normal distribution from which the noise is sampled.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    scale = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0, 15),
        description='Standard deviation of the normal distribution that generates the noise. Must be >=0. If 0 then loc will simply be added to all pixels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    per_channel = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=False,
        description='Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class AdditiveGaussianNoisePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Add noise sampled from gaussian distributions elementwise to images.
    """

    metadata = construct_primitive_metadata("augmentation", "arithmetic_AdditiveGaussianNoise")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
    
        scale = self.hyperparams["scale"]
        loc = self.hyperparams['loc']
        per_channel = self.hyperparams["per_channel"]
        seed = self.hyperparams["seed"]
        return iaa.AdditiveGaussianNoise(scale=scale, loc=loc,per_channel=per_channel, seed=seed)
