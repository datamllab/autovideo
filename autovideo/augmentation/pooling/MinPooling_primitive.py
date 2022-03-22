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

__all__ = ('MinPoolingPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    kernel_size = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(1,5),
        description='The kernel size of the pooling operation.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    keep_size = hyperparams.Hyperparameter[bool](
        default=True,
        description='After pooling, the result image will usually have a different height/width compared to the original input image. If this parameter is set to True, then the pooled image will be resized to the input image’s size, i.e. the augmenter’s output shape is always identical to the input shape.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class MinPoolingPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply min pooling to images.
    """

    metadata = construct_primitive_metadata("augmentation", "pooling_MinPooling")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        keep_size = self.hyperparams['keep_size']
        kernel_size = self.hyperparams["kernel_size"]
        return iaa.MinPooling(kernel_size=kernel_size,keep_size=keep_size,seed=seed)