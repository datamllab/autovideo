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

__all__ = ('LinearContrastPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    alpha = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.6, 1.4),
        description="Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) the difference between each pixel value and the dtype’s center value, e.g. 127 for uint8.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    per_channel = hyperparams.Hyperparameter[bool](
        default=False,
        description='Whether to use the same factor for all channels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class LinearContrastPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Adjust contrast by scaling each pixel to 127 + alpha*(v-127).
    """

    metadata = construct_primitive_metadata("augmentation", "contrast_LinearContrast")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        alpha = self.hyperparams["alpha"]
        per_channel = self.hyperparams["per_channel"]
        seed = self.hyperparams["seed"]
        return iaa.LinearContrast(alpha=alpha, per_channel=per_channel, seed=seed)

