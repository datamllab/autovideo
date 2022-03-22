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

__all__ = ('FastSnowyLandscapePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    lightness_threshold  = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=140,
        description='All pixels with lightness in HLS colorspace that is below this value will have their lightness increased by lightness_multiplier.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    lightness_multiplier = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=2.5,
        description='Multiplier for pixel lightness value in HLS colorspace. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class FastSnowyLandscapePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which convert non-snowy landscapes to snowy ones.
    """

    metadata = construct_primitive_metadata("augmentation", "weather_FastSnowyLandscape")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        lightness_threshold = self.hyperparams["lightness_threshold"]
        lightness_multiplier = self.hyperparams["lightness_multiplier"]
        return iaa.FastSnowyLandscape(lightness_threshold = lightness_threshold,lightness_multiplier=lightness_multiplier, seed=seed)
