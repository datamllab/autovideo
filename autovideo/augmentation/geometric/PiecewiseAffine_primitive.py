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

__all__ = ('PiecewiseAffinePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    scale = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.01, 0.05),
        description="Each point on the regular grid is moved around via a normal distribution. This scale factor is equivalent to the normal distribution’s sigma. Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of the image if absolute_scale=False (default), so this scale can be the same for different sized images. Recommended values are in the range 0.01 to 0.05 (weak to strong augmentations).",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    nb_rows = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=8,
        description="Number of rows of points that the regular grid should have. Must be at least 2. For large images, you might want to pick a higher value than 4. You might have to then adjust scale to lower values.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    nb_cols = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=8,
        description="Number of columns. Analogous to nb_rows",
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

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class PiecewiseAffinePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply affine transformations that differ between local neighbourhoods.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_PiecewiseAffine")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        scale = self.hyperparams["scale"]
        nb_rows = self.hyperparams["nb_rows"]
        nb_cols = self.hyperparams["nb_cols"]
        order = self.hyperparams["order"]
        cval = self.hyperparams["cval"]
        mode = self.hyperparams["mode"]
        seed = self.hyperparams["seed"]
        return iaa.PiecewiseAffine(scale=scale, nb_rows=nb_rows, nb_cols=nb_cols, seed=seed, order=order, cval=cval, mode=mode)

