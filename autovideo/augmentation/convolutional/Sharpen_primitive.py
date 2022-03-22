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

__all__ = ('SharpenPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    alpha = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.0, 0.2),
        description="Blending factor of the sharpened image. At 0.0, only the original image is visible, at 1.0 only its sharpened version is visible",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    lightness = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.8, 1.2),
        description="Lightness/brightness of the sharped image. Sane values are somewhere in the interval [0.5, 2.0]. The value 0.0 results in an edge map. Values higher than 1.0 create bright images. Default value is 1.0.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class SharpenPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Augmenter that sharpens images and overlays the result with the original image.
    """

    metadata = construct_primitive_metadata("augmentation", "convolutional_Sharpen")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        alpha = self.hyperparams["alpha"]
        lightness = self.hyperparams["lightness"]
        seed = self.hyperparams["seed"]
        return iaa.Sharpen(alpha=alpha, lightness=lightness, seed=seed)

