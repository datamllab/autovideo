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

__all__ = ('CutoutPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    nb_iterations = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=1,
        description='How many rectangular areas to fill.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    position = hyperparams.Enumeration(
        values=['uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'],
        default='uniform',
        description="Defines the position of each area to fill.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    size = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=0.2,
        description='The size of the rectangle to fill as a fraction of the corresponding image size,',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    squared = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=True,
        description='Whether to generate only squared areas cutout areas or allow rectangular ones. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    fill_mode = hyperparams.Hyperparameter[typing.Union[str,list]](
        default='constant',
        description='Mode to use in order to fill areas',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=128,
        description='The value to use (i.e. the color) to fill areas if fill_mode is `constant.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    fill_per_channel = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=False,
        description='Whether to fill each area in a channelwise fashion (True) or not (False).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class CutoutPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Fill one or more rectangular areas in an image using a fill mode.
    """

    metadata = construct_primitive_metadata("augmentation", "arithmetic_Cutout")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
    
        nb_iterations = self.hyperparams["nb_iterations"]
        position = self.hyperparams["position"]
        size = self.hyperparams['size']
        squared = self.hyperparams['squared']
        fill_mode = self.hyperparams['fill_mode']
        cval = self.hyperparams['cval']
        seed = self.hyperparams["seed"]
        fill_per_channel = self.hyperparams['fill_per_channel']
        return iaa.Cutout(nb_iterations=nb_iterations, position=position, size=size,squared=squared,fill_mode=fill_mode,cval=cval,fill_per_channel=fill_per_channel,seed=seed)