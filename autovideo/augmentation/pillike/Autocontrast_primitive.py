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

__all__ = ('AutocontrastPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    cutoff = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(10,20),
        description='Percentage of values to cut off from the low and high end of each image’s histogram, before stretching it to [0, 255]',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    per_channel = hyperparams.Hyperparameter[typing.Union[bool,float]](
        default=False,
        description='Whether to use the same value for all channels (False) or to sample a new value for each channel (True)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class AutocontrastPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Adjust contrast by cutting off p% of lowest/highest histogram values.
    """

    metadata = construct_primitive_metadata("augmentation", "pillike_Autocontrast")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        cutoff = self.hyperparams['cutoff']
        per_channel = self.hyperparams['per_channel']
        seed = self.hyperparams["seed"]
        return iaa.Autocontrast(cutoff=cutoff,per_channel=per_channel,seed=seed)