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

__all__ = ('SnowflakesPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    density  = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.01, 0.075),
        description= "Density of the snowflake layer, as a probability of each pixel in low resolution space to be a snowflake.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    density_uniformity = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.45, 0.55),
        description='Size uniformity of the snowflakes. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    flake_size = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.2, 0.7),
        description='This parameter controls the resolution at which snowflakes are sampled.  ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    flake_size_uniformity = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.45, 0.55),
        description=' Controls the size uniformity of the snowflakes.  ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    angle = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(-30,30),
        description='Angle in degrees of motion blur applied to the snowflakes  ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    speed = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.007, 0.03),
        description='Perceived falling speed of the snowflakes.   ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class SnowflakesPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which add falling snowflakes to images.
    """

    metadata = construct_primitive_metadata("augmentation", "weather_Snowflakes")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        density = self.hyperparams["density"]
        density_uniformity = self.hyperparams["density_uniformity"]
        flake_size = self.hyperparams["flake_size"]
        flake_size_uniformity = self.hyperparams["flake_szie_uniformity"]
        angle = self.hyperparams["angle"]
        speed = self.hyperparams["speed"]
        return iaa.Snowflakes(density = density,density_uniformity = density_uniformity,flake_size = flake_size,flake_size_uniformity = flake_szie_uniformity,angle = angle,speed = speed, seed=seed)
