# List of Augmentation Primitives


## Arithmetic
| Augmentation Method             | Primitive Path                                                                                                                                           |
| :-----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AddElementwise                  | [autovideo/augmentation/arithmetic/AddElementwise_primitive.py](../autovideo/augmentation/arithmetic/AddElementwise_primitive.py)                        |
| Add                             | [autovideo/augmentation/arithmetic/Add_primitive.py](../autovideo/augmentation/arithmetic/Add_primitive.py)                                              |
| AdditiveGaussianNoise           | [autovideo/augmentation/arithmetic/AdditiveGaussianNoise_primitive.py](../autovideo/augmentation/arithmetic/AdditiveGaussianNoise_primitive.py)          |
| AdditiveLaplaceNoise            | [autovideo/augmentation/arithmetic/AdditiveLaplaceNoise_primitive.py](../autovideo/augmentation/arithmetic/AdditiveLaplaceNoise_primitive.py)            |
| CoarseDropout                   | [autovideo/augmentation/arithmetic/CoarseDropout_primitive.py](../autovideo/augmentation/arithmetic/CoarseDropout_primitive.py)                          |
| CoarsePepper                    | [autovideo/augmentation/arithmetic/CoarsePepper_primitive.py](../autovideo/augmentation/arithmetic/CoarsePepper_primitive.py)                            |
| CoarseSaltAndPepper             | [autovideo/augmentation/arithmetic/CoarseSaltAndPepper_primitive.py](../autovideo/augmentation/arithmetic/CoarseSaltAndPepper_primitive.py)              |
| CoarseSalt                      | [autovideo/augmentation/arithmetic/CoarseSalt_primitive.py](../autovideo/augmentation/arithmetic/CoarseSalt_primitive.py)                                |
| Dropout2D                       | [autovideo/augmentation/arithmetic/Dropout2D_primitive.py](../autovideo/augmentation/arithmetic/Dropout2D_primitive.py)                                  |
| Dropout                         | [autovideo/augmentation/arithmetic/Dropout_primitive.py](../autovideo/augmentation/arithmetic/Dropout_primitive.py)                                      |
| ImpulseNoise                    | [autovideo/augmentation/arithmetic/ImpulseNoise_primitive.py](../autovideo/augmentation/arithmetic/ImpulseNoise_primitive.py)                            |
| Invert                          | [autovideo/augmentation/arithmetic/Invert_primitive.py](../autovideo/augmentation/arithmetic/Invert_primitive.py)                                        |
| JpegCompression                 | [autovideo/augmentation/arithmetic/JpegCompression_primitive.py](../autovideo/augmentation/arithmetic/JpegCompression_primitive.py)                      |
| MultiplyElementwise             | [autovideo/augmentation/arithmetic/MultiplyElementwise_primitive.py](../autovideo/augmentation/arithmetic/MultiplyElementwise_primitive.py)              |
| Multiply                        | [autovideo/augmentation/arithmetic/Multiply_primitive.py](../autovideo/augmentation/arithmetic/Multiply_primitive.py)                                    |
| Pepper                          | [autovideo/augmentation/arithmetic/Pepper_primitive.py](../autovideo/augmentation/arithmetic/Pepper_primitive.py)                                        |
| ReplaceElementwise              | [autovideo/augmentation/arithmetic/ReplaceElementwise_primitive.py](../autovideo/augmentation/arithmetic/ReplaceElementwise_primitive.py)                |
| SaltAndPepper                   | [autovideo/augmentation/arithmetic/SaltAndPepper_primitive.py](../autovideo/augmentation/arithmetic/SaltAndPepper_primitive.py)                          |
| Salt                            | [autovideo/augmentation/arithmetic/Salt_primitive.py](../autovideo/augmentation/arithmetic/Salt_primitive.py)                                            |
| Solarize                        | [autovideo/augmentation/arithmetic/Solarize_primitive.py](../autovideo/augmentation/arithmetic/Solarize_primitive.py)                                    |
| TotalDropout                    | [autovideo/augmentation/arithmetic/TotalDropout_primitive.py](../autovideo/augmentation/arithmetic/TotalDropout_primitive.py)                            |

## Artistic
| Augmentation Method             | Primitive Path                                                                                                                                           |
| :-----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Cartoon                         | [autovideo/augmentation/artistic/Cartoon_primitive.py](../autovideo/augmentation/artistic/Cartoon_primitive.py)                                          |

## Blend
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| BlendAlphaBoundingBoxes            | [autovideo/augmentation/blend/BlendAlphaBoundingBoxes_primitive.py](../autovideo/augmentation/artistic/BlendAlphaBoundingBoxes_primitive.py)                            |
| BlendAlphaCheckerboard             | [autovideo/augmentation/blend/BlendAlphaCheckerboard_primitive.py](../autovideo/augmentation/artistic/BlendAlphaCheckerboard_primitive.py)                              |
| BlendAlphaElementwise              | [autovideo/augmentation/blend/BlendAlphaElementwise_primitive.py](../autovideo/augmentation/artistic/BlendAlphaElementwise_primitive.py)                                |
| BlendAlphaFrequencyNoise           | [autovideo/augmentation/blend/BlendAlphaFrequencyNoise_primitive.py](../autovideo/augmentation/artistic/BlendAlphaFrequencyNoise_primitive.py)                          |
| BlendAlphaHorizontalLinearGradient | [autovideo/augmentation/blend/BlendAlphaHorizontalLinearGradient_primitive.py](../autovideo/augmentation/artistic/BlendAlphaHorizontalLinearGradient_primitive.py)      |
| BlendAlphaRegularGrid              | [autovideo/augmentation/blend/BlendAlphaRegularGrid_primitive.py](../autovideo/augmentation/artistic/BlendAlphaRegularGrid_primitive.py)                                |
| BlendAlphaSegMapClassIds           | [autovideo/augmentation/blend/BlendAlphaSegMapClassIds_primitive.py](../autovideo/augmentation/artistic/BlendAlphaSegMapClassIds_primitive.py)                          |
| BlendAlphaSimplexNoise             | [autovideo/augmentation/blend/BlendAlphaSimplexNoise_primitive.py](../autovideo/augmentation/artistic/BlendAlphaSimplexNoise_primitive.py)                              |
| BlendAlphaSomeColors               | [autovideo/augmentation/blend/BlendAlphaSomeColors_primitive.py](../autovideo/augmentation/artistic/BlendAlphaSomeColors_primitive.py)                                  |
| BlendAlphaSomeColors               | [autovideo/augmentation/blend/BlendAlphaSomeColors_primitive.py](../autovideo/augmentation/artistic/BlendAlphaSomeColors_primitive.py)                                  |
| BlendAlphaVerticalLinearGradient   | [autovideo/augmentation/blend/BlendAlphaVerticalLinearGradient_primitive.py](../autovideo/augmentation/artistic/BlendAlphaVerticalLinearGradient_primitive.py)          |
| BlendAlpha                         | [autovideo/augmentation/blend/BlendAlpha_primitive.py](../autovideo/augmentation/artistic/BlendAlpha_primitive.py)                                                      |

## Blur
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AverageBlur                        | [autovideo/augmentation/blend/AverageBlur_primitive.py](../autovideo/augmentation/blur/AverageBlur_primitive.py)                                                           |
| BilateralBlur                      | [autovideo/augmentation/blend/BilateralBlur_primitive.py](../autovideo/augmentation/blur/BilateralBlur_primitive.py)                                                       |
| GaussianBlur                       | [autovideo/augmentation/blend/GaussianBlur_primitive.py](../autovideo/augmentation/blur/GaussianBlur_primitive.py)                                                         |
| MeanShiftBlur                      | [autovideo/augmentation/blend/MeanShiftBlur_primitive.py](../autovideo/augmentation/blur/MeanShiftBlur_primitive.py)                                                       |
| MedianBlur                         | [autovideo/augmentation/blend/MedianBlur_primitive.py](../autovideo/augmentation/blur/MedianBlur_primitive.py)                                                             |
| MotionBlur                         | [autovideo/augmentation/blend/MotionBlur_primitive.py](../autovideo/augmentation/blur/MotionBlur_primitive.py)                                                             |

## Collections
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RandAugment                        | [autovideo/augmentation/collections/RandAugment_primitive.py](../autovideo/augmentation/collections/RandAugment_primitive.py)                                              |

## Color
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AddToBrightness                    | [autovideo/augmentation/color/AddToBrightness_primitive.py](../autovideo/augmentation/color/AddToBrightness_primitive.py)                                                  |
| AddToHueAndSaturation              | [autovideo/augmentation/color/AddToHueAndSaturation_primitive.py](../autovideo/augmentation/color/AddToHueAndSaturation_primitive.py)                                      |
| AddToHue                           | [autovideo/augmentation/color/AddToHue_primitive.py](../autovideo/augmentation/color/AddToHue_primitive.py)                                                                |
| AddToSaturation                    | [autovideo/augmentation/color/AddToSaturation_primitive.py](../autovideo/augmentation/color/AddToSaturation_primitive.py)                                                  |
| ChangeColorTemperature             | [autovideo/augmentation/color/ChangeColorTemperature_primitive.py](../autovideo/augmentation/color/ChangeColorTemperature_primitive.py)                                    |
| ChangeColorspace                   | [autovideo/augmentation/color/ChangeColorspace_primitive.py](../autovideo/augmentation/color/ChangeColorspace_primitive.py)                                                |
| Grayscale                          | [autovideo/augmentation/color/Grayscale_primitive.py](../autovideo/augmentation/color/Grayscale_primitive.py)                                                              |
| KMeansColorQuantization            | [autovideo/augmentation/color/KMeansColorQuantization_primitive.py](../autovideo/augmentation/color/KMeansColorQuantization_primitive.py)                                  |
| MultiplyBrightness                 | [autovideo/augmentation/color/MultiplyBrightness_primitive.py](../autovideo/augmentation/color/MultiplyBrightness_primitive.py)                                            |
| MultiplyHueAndSaturation           | [autovideo/augmentation/color/MultiplyHueAndSaturation_primitive.py](../autovideo/augmentation/color/MultiplyHueAndSaturation_primitive.py)                                |
| MultiplyHue                        | [autovideo/augmentation/color/MultiplyHue_primitive.py](../autovideo/augmentation/color/MultiplyHue_primitive.py)                                                          |
| MultiplySaturation                 | [autovideo/augmentation/color/MultiplySaturation_primitive.py](../autovideo/augmentation/color/MultiplySaturation_primitive.py)                                            |
| Posterize                          | [autovideo/augmentation/color/Posterize_primitive.py](../autovideo/augmentation/color/Posterize_primitive.py)                                                              |
| RemoveSaturation                   | [autovideo/augmentation/color/RemoveSaturation_primitive.py](../autovideo/augmentation/color/RemoveSaturation_primitive.py)                                                |
| UniformColorQuantizationToNBits    | [autovideo/augmentation/color/UniformColorQuantizationToNBits_primitive.py](../autovideo/augmentation/color/UniformColorQuantizationToNBits_primitive.py)                  |
| UniformColorQuantization_primitive | [autovideo/augmentation/color/UniformColorQuantization_primitive_primitive.py](../autovideo/augmentation/color/UniformColorQuantization_primitive_primitive.py)            |

## Contrast
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AllChannelsCLAHE                   | [autovideo/augmentation/contrast/AllChannelsCLAHE_primitive.py](../autovideo/augmentation/color/AllChannelsCLAHE_primitive.py)                                             |
| AllChannelsHistogramEqualization   | [autovideo/augmentation/contrast/AllChannelsHistogramEqualization_primitive.py](../autovideo/augmentation/color/AllChannelsHistogramEqualization_primitive.py)             |
| CLAHE                              | [autovideo/augmentation/contrast/CLAHE_primitive.py](../autovideo/augmentation/color/CLAHE_primitive.py)                                                                   |
| GammaContrast                      | [autovideo/augmentation/contrast/GammaContrast_primitive.py](../autovideo/augmentation/color/GammaContrast_primitive.py)                                                   |
| HistogramEqualization              | [autovideo/augmentation/contrast/HistogramEqualization_primitive.py](../autovideo/augmentation/color/HistogramEqualization_primitive.py)                                   |
| LinearContrast                     | [autovideo/augmentation/contrast/LinearContrast_primitive.py](../autovideo/augmentation/color/LinearContrast_primitive.py)                                                 |
| LogContrast                        | [autovideo/augmentation/contrast/LogContrast_primitive.py](../autovideo/augmentation/color/LogContrast_primitive.py)                                                       |
| SigmoidContrast                    | [autovideo/augmentation/contrast/SigmoidContrast_primitive.py](../autovideo/augmentation/color/SigmoidContrast_primitive.py)                                               |

## Convolutional
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DirectedEdgeDetect                 | [autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py](../autovideo/augmentation/convolutional/DirectedEdgeDetect_primitive.py)                            |
| EdgeDetect                         | [autovideo/augmentation/convolutional/EdgeDetect_primitive.py](../autovideo/augmentation/convolutional/EdgeDetect_primitive.py)                                            |
| Emboss                             | [autovideo/augmentation/convolutional/Emboss_primitive.py](../autovideo/augmentation/convolutional/Emboss_primitive.py)                                                    |
| Sharpen                            | [autovideo/augmentation/convolutional/Sharpen_primitive.py](../autovideo/augmentation/convolutional/Sharpen_primitive.py)                                                  |

## Debug
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SaveDebugImageEveryNBatches        | [autovideo/augmentation/edges/SaveDebugImageEveryNBatches_primitive.py](../autovideo/augmentation/edges/SaveDebugImageEveryNBatches_primitive.py)                          |

## Edges
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Canny                              | [autovideo/augmentation/debug/Canny_primitive.py](../autovideo/augmentation/debug/Canny_primitive.py)                                                                      |

## Flip
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Fliplr                             | [autovideo/augmentation/debug/Fliplr_primitive.py](../autovideo/augmentation/debug/Fliplr_primitive.py)                                                                    |
| Flipud                             | [autovideo/augmentation/debug/Fliplr_primitive.py](../autovideo/augmentation/debug/Fliplr_primitive.py)                                                                    |
| HorizontalFlip                     | [autovideo/augmentation/debug/HorizontalFlip_primitive.py](../autovideo/augmentation/debug/HorizontalFlip_primitive.py)                                                    |
| VericalFlip                        | [autovideo/augmentation/debug/VericalFlip_primitive.py](../autovideo/augmentation/debug/VericalFlip_primitive.py)                                                          |


