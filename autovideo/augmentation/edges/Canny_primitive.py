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

__all__ = ('CannyPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    alpha = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.0, 1.0),
        description="Blending factor to use in alpha blending. A value close to 1.0 means that only the edge image is visible. A value close to 0.0 means that only the original image is visible. A value close to 0.5 means that the images are merged according to 0.5*image + 0.5*edge_image. If a sample from this parameter is 0, no action will be performed for the corresponding image.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    hysteresis_thresholds = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(60, 140),
        description="Min and max values to use in hysteresis thresholding.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    sobel_kernel_size = hyperparams.Hyperparameter[typing.Union[int,tuple,list]](
        default=(3, 7),
        description="Kernel size of the sobel operator initially applied to each image. This corresponds to apertureSize in cv2.Canny(). If a sample from this parameter is <=1, no action will be performed for the corresponding image. The maximum for this parameter is 7 (inclusive). Higher values are not accepted by OpenCV. If an even value v is sampled, it is automatically changed to v-1.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class CannyPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply a canny edge detector to input images.
    """

    metadata = construct_primitive_metadata("augmentation", "edges_Canny")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        alpha = self.hyperparams["alpha"]
        sobel_kernel_size = self.hyperparams["sobel_kernel_size"]
        hysteresis_thresholds = self.hyperparams['hysteresis_thresholds']
        seed = self.hyperparams["seed"]
        return iaa.Canny(alpha=alpha, hysteresis_thresholds=hysteresis_thresholds ,sobel_kernel_size=sobel_kernel_size, seed=seed)

