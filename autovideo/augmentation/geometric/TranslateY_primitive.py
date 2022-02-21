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

__all__ = ('TranslateYPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    px = hyperparams.Set[int](
        default=(-20, 20),
        description="Analogous to translate_px in Affine, except that this translation value only affects the x-axis. No dictionary input is allowed.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    percent = hyperparams.Set[float](
        default=(-0.1, 0.1),
        description="Analogous to translate_percent in Affine, except that this translation value only affects the x-axis. No dictionary input is allowed.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class TranslateYPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply affine translation on the y-axis to input data.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_TranslateY")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        px = self.hyperparams["px"]
        percent = self.hyperparams["percent"]
        seed = self.hyperparams["seed"]
        return iaa.TranslateY(px=px, percent=percent, seed=seed)

