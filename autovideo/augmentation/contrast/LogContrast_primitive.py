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

__all__ = ('LogContrastPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    gain = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.4, 1.6),
        description="Multiplier for the logarithm result. Values around 1.0 lead to a contrast-adjusted images. Values above 1.0 quickly lead to partially broken images due to exceeding the datatype’s value range.",
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



class LogContrastPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Adjust image contrast by scaling pixels to 255*gain*log_2(1+v/255).
    """

    metadata = construct_primitive_metadata("augmentation", "contrast_LogContrast")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        gain = self.hyperparams["gain"]
        per_channel = self.hyperparams["per_channel"]
        seed = self.hyperparams["seed"]
        return iaa.LogContrast(gain=gain, per_channel=per_channel, seed=seed)

