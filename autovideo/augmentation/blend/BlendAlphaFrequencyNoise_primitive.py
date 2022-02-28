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

__all__ = ('BlendAlphaFrequencyNoisePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    per_channel = hyperparams.Constant[bool](
        default=True,
        description='Whether to use the same factor for all channels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    exponent = hyperparams.Constant[int](
        default=-2,
        description='Exponent to use when scaling in the frequency domain',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    sigmoid = hyperparams.Constant[bool](
        default=False,
        description='Whether to apply a sigmoid function to the final noise maps',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    upscale_method = hyperparams.Enumeration(
        values=['linear', 'nearest', 'cubic', 'area'],
        default='linear',
        description="After generating the noise maps in low resolution environments, they have to be upscaled to the input image size. This parameter controls the upscaling method.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    

class BlendAlphaFrequencyNoisePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Alpha-blend two image sources using simplex noise alpha masks.
    """

    metadata = construct_primitive_metadata("augmentation", "blend_BlendAlphaFrequencyNoise")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        whether_channel = self.hyperparams["per_channel"]
        seed = self.hyperparams["seed"]
        sigmoid = self.hyperparams["sigmoid"]
        upscale_method = self.hyperparams["upscale_method"]
        exponent = self.hyperparams['exponent']
        return iaa.BlendAlphaFrequencyNoise(per_channel=whether_channel, sigmoid=sigmoid, exponent=exponent, upscale_method=upscale_method, seed=seed)

