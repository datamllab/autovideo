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

__all__ = ('InvertPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    
    p = hyperparams.Hyperparameter[float](
        default=0.5,
        description='The probability of an image to be inverted.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    per_channel = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=False,
        description='Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    min_value = hyperparams.Hyperparameter[typing.Union[float,None]](
        default=None,
        description='Minimum of the value range of input images',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    max_value = hyperparams.Hyperparameter[typing.Union[float,None]](
        default=None,
        description='Maximum of the value range of input images',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    threshold = hyperparams.Hyperparameter[typing.Union[float,tuple,list,None]](
        default=None,
        description=' A threshold to use in order to invert only numbers above or below the threshold.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    invert_above_threshold = hyperparams.Hyperparameter[typing.Union[float,bool]](
        default=0.5,
        description=' If True, only values >=threshold will be inverted. Otherwise, only values <threshold will be inverted.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int]( 
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class InvertPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Augmenter that inverts all values in images, i.e. sets a pixel from value v to 255-v.
    """

    metadata = construct_primitive_metadata("augmentation", "arithmetic_Invert")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        p = self.hyperparams["p"]
        per_channel = self.hyperparams['per_channel']
        min_value = self.hyperparams['min_value']
        max_value = self.hyperparams['max_value']
        threshold = self.hyperparams["threshold"]
        invert_above_threshold = self.hyperparams['invert_above_threshold']
        return iaa.Invert(p=p,threshold=threshold,seed=seed,min_value=min_value,max_value=max_value,invert_above_threshold=invert_above_threshold,per_channel=per_channel)