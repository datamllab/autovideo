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

from autovideo.utils import construct_primitive_metadata
from autovideo.base.augmentation_base import AugmentationPrimitiveBase

__all__ = ('BlendAlphaHorizontalLinearGradientPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    min_value = hyperparams.Constant[float](
        default=0.2,
        description='Opacity of the results of the foreground branch.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    max_value = hyperparams.Constant[float](
        default=0.8,
        description='Opacity of the results of the foreground branch.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    start = hyperparams.Set[float](
        default=(0.0, 1.0),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    end = hyperparams.Set[float](
        default=(0.0, 1.0),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class BlendAlphaHorizontalLinearGradientPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Blend images from two branches along a horizontal linear gradient.
    """

    metadata = construct_primitive_metadata("augmentation", "blend_BlendAlphaHorizontalLinearGradient")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        min_ = self.hyperparams["min_value"]
        max_ = self.hyperparams["max_value"]
        seed = self.hyperparams["seed"]
        start = self.hyperparams["start"]
        end = self.hyperparams["end"]
        return iaa.BlendAlphaHorizontalLinearGradient(start_at=start, end_at=end, min_value = min_, max_value = max_, seed=seed)

