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

__all__ = ('PerspectiveTransformPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    scale = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.01, 0.15),
        description="Standard deviation of the normal distributions. These are used to sample the random distances of the subimage’s corners from the full image’s corners. The sampled values reflect percentage values (with respect to image height/width). Recommended values are in the range 0.0 to 0.1.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    keep_size = hyperparams.Hyperparameter[bool](
        default=False,
        description='Whether to resize image’s back to their original size after applying the perspective transform. If set to False, the resulting images may end up having different shapes and will always be a list, never an array.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0,255),
        description=" The constant value to use when filling in newly created pixels.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class PerspectiveTransformPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply random four point perspective transformations to images.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_PerspectiveTransform")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        scale = self.hyperparams["scale"]
        keep_size = self.hyperparams["keep_size"]
        cval = self.hyperparams["cval"]
        seed = self.hyperparams["seed"]
        return iaa.PerspectiveTransform(scale=scale, keep_size=keep_size, seed=seed, cval=cval)

