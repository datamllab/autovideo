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

__all__ = ('CartoonPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    blur_ksize = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(1, 5),
        description='Median filter kernel size.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    segmentation_ksize = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.8, 1.2),
        description=' Mean-Shift segmentation size multiplier.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    saturation = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(1.5, 2.5),
        description='Saturation multiplier.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    edge_prevalence = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.9, 1.1),
        description=' Multiplier for the prevalence of edges. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class CartoonPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Convert the style of images to a more cartoonish one.
    """

    metadata = construct_primitive_metadata("augmentation", "artistic_Cartoon")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        blur_ksize = self.hyperparams['blur_ksize']
        segmentation_ksize = self.hyperparams['segmentation_ksize']
        saturation = self.hyperparams['saturation']
        edge_prevalence = self.hyperparams['edge_prevalence']
        seed = self.hyperparams["seed"]
        return iaa.Cartoon(blur_ksize=blur_ksize,segmentation_ksize=segmentation_ksize,saturation=saturation,edge_prevalence=edge_prevalence,seed=seed)

