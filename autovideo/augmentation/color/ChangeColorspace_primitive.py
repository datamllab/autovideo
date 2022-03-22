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

__all__ = ('ChangeColorspacePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    to_colorspace = hyperparams.Enumeration(
        values=['RGB', 'BGR', 'GRAY', 'CIE', 'YCrCb', 'HSV', 'HLS', 'Lab', 'Luv'],
        default='HSV',
        description="The target colorspace. ",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    from_colorspace = hyperparams.Enumeration(
        values=['RGB', 'BGR', 'GRAY', 'CIE', 'YCrCb', 'HSV', 'HLS', 'Lab', 'Luv'],
        default='RGB',
        description="The source colorspace. ",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    alpha = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default= 1.0,
        description='the alpha value of the new colorspace when overlayed over the old one. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class ChangeColorspacePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Augmenter to change the colorspace of images..
    """

    metadata = construct_primitive_metadata("augmentation", "color_ChangeColorspace")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        to_colorspace = self.hyperparams['to_colorspace']
        from_colorspace = self.hyperparams['from_colorspace']
        alpha = self.hyperparams['alpha']
        seed = self.hyperparams["seed"]
        return iaa.ChangeColorspace(from_colorspace=from_colorspace, to_colorspace=to_colorspace, alpha=alpha,seed=seed)

