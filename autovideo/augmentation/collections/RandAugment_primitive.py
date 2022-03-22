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

__all__ = ('RandAugmentPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    n = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=2,
        description='number of transformations to apply.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    m = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(6,12),
        description='magnitude/severity/strength of the applied transformations in interval [0 .. 30] with M=0 being the weakest.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=128,
        description='The constant value to use when filling in newly created pixels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class RandAugmentPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply RandAugment to inputs as described in the corresponding paper..
    """

    metadata = construct_primitive_metadata("augmentation", "blur_RandAugment")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        n = self.hyperparams["n"]
        m = self.hyperparams["m"]
        cval = self.hyperparams['cval']
        seed = self.hyperparams["seed"]
        return iaa.RandAugment(n=n, m=m, cval=cval,seed=seed)

