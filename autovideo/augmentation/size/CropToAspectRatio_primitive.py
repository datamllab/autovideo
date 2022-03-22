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

__all__ = ('CropToAspectRatioPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    aspect_ratio = hyperparams.Hyperparameter[float](
        default=2.0,
        description='The desired aspect ratio, given as width/height',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    position = hyperparams.Enumeration(
        values=['uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'],
        default='uniform',
        description=" Sets the center point of the padding, which determines how the required padding amounts are distributed to each side. ",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class CropToAspectRatioPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Crop images until their width/height matches an aspect ratio.
    """

    metadata = construct_primitive_metadata("augmentation", "size_CropToAspectRatio")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        aspect_ratio = self.hyperparams["aspect_ratio"]
        position = self.hyperparams["position"]
        return iaa.CropToAspectRatio(aspect_ratio = aspect_ratio,position = position, seed=seed)
