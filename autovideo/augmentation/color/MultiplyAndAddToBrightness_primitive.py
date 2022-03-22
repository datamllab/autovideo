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

__all__ = ('MultiplyAndAddToBrightnessPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    mul = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.7, 1.3),
        description='Multiply.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    add = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(-30,30),
        description='Add.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    random_order = hyperparams.Hyperparameter[bool](
        default=True,
        description='Whether to apply the add and multiply operations in random order (True). ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class MultiplyAndAddToBrightnessPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Multiply and add to the brightness channels of input images.
    """

    metadata = construct_primitive_metadata("augmentation", "color_MultiplyAndAddToBrightness")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        mul = self.hyperparams["mul"]
        add = self.hyperparams["add"]
        random_order = self.hyperparams['random_order']
        seed = self.hyperparams["seed"]
        return iaa.MultiplyAndAddToBrightness(mul=mul, add=add,random_order=random_order, seed=seed)

