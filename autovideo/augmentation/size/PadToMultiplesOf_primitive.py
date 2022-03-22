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

__all__ = ('PadToMultiplesOfPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    width_multiple = hyperparams.Hyperparameter[typing.Union[int,None]](
        default=10,
        description='Pad images up to this minimum width.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    height_multiple = hyperparams.Hyperparameter[typing.Union[int,None]](
        default=6,
        description='Pad images up to this minimum height.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    position = hyperparams.Enumeration(
        values=['uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'],
        default='uniform',
        description=" Sets the center point of the padding, which determines how the required padding amounts are distributed to each side. ",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class PadToMultiplesOfPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Pad images until their height/width is a multiple of a value.
    """

    metadata = construct_primitive_metadata("augmentation", "size_PadToMultiplesOf")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        height_multiple = self.hyperparams["height_multiple"]
        width_multiple = self.hyperparams["width_multiple"]
        position = self.hyperparams["position"]
        return iaa.PadToMultiplesOf(width_multiple = width_multiple,height_multiple=height_multiple,position = position, seed=seed)
