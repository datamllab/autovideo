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

from autovideo.utils import construct_primitive_metadata
from autovideo.base.augmentation_base import AugmentationPrimitiveBase

__all__ = ('CenterCropToMultiplesOfPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    width_multiple = hyperparams.Hyperparameter[int](
        default=10,
        description='Crop images up to this base of width.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    height_multiple = hyperparams.Hyperparameter[int](
        default=6,
        description='Crop images up to this base of height.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class CenterCropToMultiplesOfPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Crop images equally on all sides until H/W are multiples of given values.
    """

    metadata = construct_primitive_metadata("augmentation", "size_CenterCropToMultiplesOf")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        height_multiple = self.hyperparams["height_multiple"]
        width_multiple = self.hyperparams["width_multiple"]
        return iaa.CenterCropToMultiplesOf(width_multiple = width_multiple,height_multiple =height_multiple, seed=seed)
