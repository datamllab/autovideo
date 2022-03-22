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

from fnmatch import translate
from d3m import container
from d3m.metadata import hyperparams
import imgaug.augmenters as iaa
import typing
from autovideo.utils import construct_primitive_metadata
from autovideo.base.augmentation_base import AugmentationPrimitiveBase

__all__ = ('AffinePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    scale = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.5, 1.5),
        description="Scaling factor to use, where 1.0 denotes “no change” and 0.5 is zoomed out to 50 percent of the original size.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    rotate = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(-45, 45),
        description="Rotation in degrees (NOT radians), i.e. expected value range is around [-360, 360]. Rotation happens around the center of the image, not the top left corner as in some other frameworks.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    shear = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(-16, 16),
        description="Shear in degrees (NOT radians), i.e. expected value range is around [-360, 360], with reasonable values being in the range of [-45, 45].",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    translate_percent = hyperparams.Hyperparameter[typing.Union[float,tuple,list,None]](
        default=0.1,
        description="Translation as a fraction of the image height/width (x-translation, y-translation), where 0 denotes “no change” and 0.5 denotes “half of the axis size”.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    order = hyperparams.Hyperparameter[typing.Union[int,list]](
        default=1,
        description="interpolation order to use",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0,255),
        description=" The constant value to use when filling in newly created pixels.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    mode = hyperparams.Hyperparameter[typing.Union[str,list]](
        default='constant',
        description="Method to use when filling in newly created pixels",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    translate_px = hyperparams.Hyperparameter[typing.Union[int,tuple,list,None]](
        default=None,
        description="Translation in pixels.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class AffinePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Augmenter to apply affine transformations to images.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_Affine")

    def _get_function(self):
        """
        set up function and parameter of functions
        """

        scale = self.hyperparams["scale"]
        shear = self.hyperparams["shear"]
        rotate = self.hyperparams["rotate"]
        translate_percent = self.hyperparams["translate_percent"]
        translate_px = self.hyperparams["translate_px"]
        order = self.hyperparams["order"]
        cval = self.hyperparams["cval"]
        mode = self.hyperparams["mode"]
        seed = self.hyperparams["seed"]
        return iaa.Affine(order=order, cval=cval, mode=mode, scale=scale, rotate=rotate, shear=shear, seed=seed, translate_percent=translate_percent,translate_px=translate_px)

