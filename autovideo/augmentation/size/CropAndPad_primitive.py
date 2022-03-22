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

__all__ = ('CropAndPadPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    px = hyperparams.Hyperparameter[typing.Union[int,tuple,None]](
        default=None,
        description='The number of pixels to crop (negative values) or pad (positive values) on each side of the image. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    percent = hyperparams.Hyperparameter[typing.Union[float,tuple,None]](
        default=(0, 0.2),
        description='The number of pixels to crop (negative values) or pad (positive values) on each side of the image given as a fraction of the image height/width. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    pad_mode = hyperparams.Enumeration(
        values=['constant','edge','linear_ramp', 'maximum', 'median', 'minimum', 'reflect', 'symmetric', 'wrap'],
        default='constant',
        description="Interpolation to use.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    pad_cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list,None]](
        default=(0, 128),
        description='The constant value to use if the pad mode is constant or the end value to use if the mode is linear_ramp.  ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    




class CropAndPadPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Crop/pad images by pixel amounts or fractions of image sizes.
    """

    metadata = construct_primitive_metadata("augmentation", "size_CropAndPad")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        return iaa.CropAndPad(px=self.hyperparams["px"],pad_cval=self.hyperparams["pad_cval"],percent=self.hyperparams["percent"],pad_mode=self.hyperparams['pad_mode'],seed=self.hyperparams["seed"])