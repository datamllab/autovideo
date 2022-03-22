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

__all__ = ('BilateralBlurPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    d = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(1, 9),
        description='High values for d lead to significantly worse performance. Values equal or less than 10 seem to be good. Use <5 for real-time applications.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    sigma_color = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(10, 250),
        description='Filter sigma in the color space with value range [1, inf). A large value of the parameter means that farther colors within the pixel neighborhood (see sigma_space) will be mixed together, resulting in larger areas of semi-equal color.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    sigma_space = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(10, 250),
        description='Filter sigma in the coordinate space with value range [1, inf). A large value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigma_color).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class BilateralBlurPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Blur/Denoise an image using a bilateral filter
    """

    metadata = construct_primitive_metadata("augmentation", "blur_BilateralBlur")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        sigma_color = self.hyperparams["sigma_color"]
        sigma_space = self.hyperparams["sigma_space"]
        d = self.hyperparams["d"]
        seed = self.hyperparams["seed"]
        return iaa.BilateralBlur(d=d, sigma_color=sigma_color, sigma_space=sigma_space, seed=seed)

