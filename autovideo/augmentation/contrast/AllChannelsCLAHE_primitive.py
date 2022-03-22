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

__all__ = ('AllChannelsCLAHEPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    clip_limit = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.1, 8),
        description=" Clipping limit. Higher values result in stronger contrast. OpenCV uses a default of 40, though values around 5 seem to already produce decent contrast.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    tile_grid_size_px = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(3, 12),
        description="See imgaug.augmenters.contrast.CLAHE",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    tile_grid_size_px_min = hyperparams.Hyperparameter[int](
        default=3,
        description="See imgaug.augmenters.contrast.CLAHE",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    per_channel = hyperparams.Hyperparameter[bool](
        default=False,
        description='Whether to use the same factor for all channels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class AllChannelsCLAHEPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    Apply CLAHE to all channels of images in their original colorspaces.
    CLAHE (Contrast Limited Adaptive Histogram Equalization) performs histogram equilization within image patches, i.e. over local neighbourhoods.
    """

    metadata = construct_primitive_metadata("augmentation", "contrast_AllChannelsCLAHE")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        clip_limit = self.hyperparams["clip_limit"]
        per_channel = self.hyperparams["per_channel"]
        tile_grid_size_px = self.hyperparams['tile_grid_size_px']
        tile_grid_size_px_min = self.hyperparams['tile_grid_size_px_min']
        seed = self.hyperparams["seed"]
        return iaa.AllChannelsCLAHE(clip_limit=clip_limit, tile_grid_size_px=tile_grid_size_px,tile_grid_size_px_min=tile_grid_size_px_min ,per_channel=per_channel, seed=seed)

