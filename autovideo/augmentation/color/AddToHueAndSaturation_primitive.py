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

__all__ = ('AddToHueAndSaturationPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    per_channel = hyperparams.Constant[bool](
        default=True,
        description='Whether to sample per image only one value from value and use it for both hue and saturation (False) or to sample independently one value for hue and one for saturation (True).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    value = hyperparams.Set[int](
        default= (-50, 50),
        description='Inverse multiplier to use for the saturation values. High values denote stronger color removal. E.g. 1.0 will remove all saturation, 0.0 will remove nothing. Expected value range is [0.0, 1.0].',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class AddToHueAndSaturationPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Increases or decreases hue and saturation by random values.
    """

    metadata = construct_primitive_metadata("augmentation", "color_AddToHueAndSaturation")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        value = self.hyperparams['value']
        per_channel = self.hyperparams['per_channel']
        seed = self.hyperparams["seed"]
        return iaa.AddToHueAndSaturation(value=value, per_channel=per_channel, seed=seed)

