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

__all__ = ('SaveDebugImageEveryNBatchesPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    destination = hyperparams.Constant[str](
        default="/autovideo/augmentation/debug",
        description='Path to a folder. The saved images will follow a filename pattern of batch_<batch_id>.png. The latest image will additionally be saved to latest.png.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    interval = hyperparams.Hyperparameter[int](
        default=100,
        description='Interval in batches. If set to N, every N th batch an image will be generated and saved, starting with the first observed batch. Note that the augmenter only counts batches that it sees. If it is executed conditionally or re-instantiated, it may not see all batches or the counter may be wrong in other ways.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class SaveDebugImageEveryNBatchesPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Visualize data in batches and save corresponding plots to a folder..
    """

    metadata = construct_primitive_metadata("augmentation", "debug_SaveDebugImageEveryNBatches")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        destination = self.hyperparams["destination"]
        interval = self.hyperparams["interval"]
        seed = self.hyperparams["seed"]
        return iaa.SaveDebugImageEveryNBatches(destination=destination, interval=interval, seed=seed)

