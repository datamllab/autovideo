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

__all__ = ('JigsawPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    nb_rows = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(3, 10),
        description="How many rows the jigsaw pattern should have.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    nb_cols = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(3, 10),
        description="How many cols the jigsaw pattern should have.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    max_steps = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(1, 5),
        description="How many steps each jigsaw cell may be moved.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class JigsawPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Move cells within images similar to jigsaw patterns.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_Jigsaw")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        nb_rows = self.hyperparams["nb_rows"]
        nb_cols = self.hyperparams["nb_cols"]
        max_steps = self.hyperparams["max_steps"]
        seed = self.hyperparams["seed"]
        return iaa.Jigsaw(nb_rows=nb_rows, nb_cols=nb_cols, max_steps=max_steps, seed=seed)

