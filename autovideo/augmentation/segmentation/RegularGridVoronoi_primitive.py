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

__all__ = ('RegularGridVoronoiPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    n_rows = hyperparams.Constant[int](
        default=10,
        description='Number of rows of coordinates to place on each image, i.e. the number of coordinates on the y-axis.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    n_cols = hyperparams.Constant[int](
        default=20,
        description='Number of columns of coordinates to place on each image, i.e. the number of coordinates on the x-axis. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class RegularGridVoronoiPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Completely or partially transform images to their superpixel representation.
    """

    metadata = construct_primitive_metadata("augmentation", "segmentation_RegularGridVoronoi")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        seed = self.hyperparams["seed"]
        n_rows = self.hyperparams["n_rows"]
        n_cols = self.hyperparams["n_cols"]
        return iaa.RegularGridVoronoi(n_cols=n_cols,n_rows = n_rows ,seed=seed)