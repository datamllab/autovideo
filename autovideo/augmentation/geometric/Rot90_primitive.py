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

__all__ = ('Rot90Primitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    k = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(1, 3),
        description="How often to rotate clockwise by 90 degrees.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    keep_size = hyperparams.Hyperparameter[bool](
        default=False,
        description='After rotation by an odd-valued k (e.g. 1 or 3), the resulting image may have a different height/width than the original image. If this parameter is set to True, then the rotated image will be resized to the input image’s size. Note that this might also cause the augmented image to look distorted.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class Rot90Primitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Rotate images clockwise by multiples of 90 degrees.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_Rot90")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        k = self.hyperparams["k"]
        keep_size = self.hyperparams["keep_size"]
        seed = self.hyperparams["seed"]
        return iaa.Rot90(k=k, keep_size=keep_size, seed=seed)

