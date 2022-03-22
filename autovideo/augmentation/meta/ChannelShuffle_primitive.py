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

__all__ = ('ChannelShufflePrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    p = hyperparams.Hyperparameter[float](
        default=0.35,
        description='Probability of shuffling channels in any given image.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    channels= hyperparams.Hyperparameter[typing.Union[list,None]](
        default=None,
        description='Which channels are allowed to be shuffled with each other.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class ChannelShufflePrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Randomize the order of channels in input images.
    """

    metadata = construct_primitive_metadata("augmentation", "meta_ChannelShuffle")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        p = self.hyperparams["p"]
        channels = self.hyperparams['channels']
        seed = self.hyperparams["seed"]
        return iaa.ChannelShuffle(p=p,channels=channels, seed=seed)

