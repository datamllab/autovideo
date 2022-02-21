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

__all__ = ('BlendAlphaSomeColorsPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    smoothness = hyperparams.Constant[float](
        default=0.0,
        description='Smoothness level.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    alpha = hyperparams.List[float](
        default=[0.0, 1.0],
        description='Each color tune is either selected (alpha of 1.0) or not selected (0.0).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    nb_bins = hyperparams.Constant[int](
        default=2,
        description='Exponent to use when scaling in the frequency domain',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class BlendAlphaSomeColorsPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Blend images from two branches using colorwise masks.
    """

    metadata = construct_primitive_metadata("augmentation", "blend_BlendAlphaSomeColors")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        smooth = self.hyperparams["smoothness"]
        nb_bins = self.hyperparams["nb_bins"]
        alpha = self.hyperparams['alpha']
        return iaa.BlendAlphaSomeColors(smoothness=smooth, alpha=alpha, nb_bins=nb_bins , seed=seed)

