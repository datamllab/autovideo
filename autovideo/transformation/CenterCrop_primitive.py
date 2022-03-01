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

from autovideo.utils import construct_primitive_metadata
from autovideo.base.transformation_base import TransformationPrimitiveBase
import torchvision
import random
import numbers

__all__ = ('CenterCropPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    size = hyperparams.Hyperparameter(
        default=(224, 224),
        description='Crop Size',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

class CenterCropPrimitive(TransformationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Add values to the pixels of images with possibly different values for neighbouring pixels.
    """

    metadata = construct_primitive_metadata("transformation", "CenterCrop")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        size = self.hyperparams["size"]
        return torchvision.transforms.CenterCrop(size)
