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

## Geometric
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Affine                             | [autovideo/augmentation/geometric/Affine_primitive.py](../autovideo/augmentation/geometric/Affine_primitive.py)                                                            |
| ElasticTransformation              | [autovideo/augmentation/geometric/ElasticTransformation_primitive.py](../autovideo/augmentation/geometric/ElasticTransformation_primitive.py)                              |
| Jigsaw                             | [autovideo/augmentation/geometric/Jigsaw_primitive.py](../autovideo/augmentation/geometric/Jigsaw_primitive.py)                                                            |
| PerspectiveTransform               | [autovideo/augmentation/geometric/PerspectiveTransform_primitive.py](../autovideo/augmentation/geometric/PerspectiveTransform_primitive.py)                                |
| PiecewiseAffine                    | [autovideo/augmentation/geometric/PiecewiseAffine_primitive.py](../autovideo/augmentation/geometric/PiecewiseAffine_primitive.py)                                          |
| Rot90                              | [autovideo/augmentation/geometric/Rot90_primitive.py](../autovideo/augmentation/geometric/Rot90_primitive.py)                                                              |
| Rotate                             | [autovideo/augmentation/geometric/Rotate_primitive.py](../autovideo/augmentation/geometric/Rotate_primitive.py)                                                            |
| ScaleX                             | [autovideo/augmentation/geometric/ScaleX_primitive.py](../autovideo/augmentation/geometric/ScaleX_primitive.py)                                                            |
| ScaleY                             | [autovideo/augmentation/geometric/ScaleY_primitive.py](../autovideo/augmentation/geometric/ScaleY_primitive.py)                                                            |
| ShearX                             | [autovideo/augmentation/geometric/ShearX_primitive.py](../autovideo/augmentation/geometric/ShearX_primitive.py)                                                            |
| ShearY                             | [autovideo/augmentation/geometric/ShearY_primitive.py](../autovideo/augmentation/geometric/ShearY_primitive.py)                                                            |
| TranslateX                         | [autovideo/augmentation/geometric/TranslateX_primitive.py](../autovideo/augmentation/geometric/ShearY_primitive.py)                                                        |
| TranslateY                         | [autovideo/augmentation/geometric/TranslateY_primitive.py](../autovideo/augmentation/geometric/TranslateY_primitive.py)                                                    |
| WithPolarWarping                   | [autovideo/augmentation/geometric/WithPolarWarping_primitive.py](../autovideo/augmentation/geometric/WithPolarWarping_primitive.py)                                        |

## Imgcorruptlike
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Brightness                         | [autovideo/augmentation/imgcorruptlike/Brightness_primitive.py](../autovideo/augmentation/imgcorruptlike/Brightness_primitive.py)                                          |
| Contrast                           | [autovideo/augmentation/imgcorruptlike/Contrast_primitive.py](../autovideo/augmentation/imgcorruptlike/Contrast_primitive.py)                                              |
| DefocusBlur                        | [autovideo/augmentation/imgcorruptlike/DefocusBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/DefocusBlur_primitive.py)                                        |
| ElasticTransform                   | [autovideo/augmentation/imgcorruptlike/ElasticTransform_primitive.py](../autovideo/augmentation/imgcorruptlike/ElasticTransform_primitive.py)                              |
| Fog                                | [autovideo/augmentation/imgcorruptlike/Fog_primitive.py](../autovideo/augmentation/imgcorruptlike/Fog_primitive.py)                                                        |
| Frost                              | [autovideo/augmentation/imgcorruptlike/Frost_primitive.py](../autovideo/augmentation/imgcorruptlike/Frost_primitive.py)                                                    |
| GaussianBlur                       | [autovideo/augmentation/imgcorruptlike/GaussianBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/GaussianBlur_primitive.py)                                      |
| GaussianNoise                      | [autovideo/augmentation/imgcorruptlike/GaussianNoise_primitive.py](../autovideo/augmentation/imgcorruptlike/GaussianNoise_primitive.py)                                    |
| GlassBlur                          | [autovideo/augmentation/imgcorruptlike/GlassBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/GlassBlur_primitive.py)                                            |
| ImpulseNoise                       | [autovideo/augmentation/imgcorruptlike/ImpulseNoise_primitive.py](../autovideo/augmentation/imgcorruptlike/ImpulseNoise_primitive.py)                                      |
| JpegCompression                    | [autovideo/augmentation/imgcorruptlike/JpegCompression_primitive.py](../autovideo/augmentation/imgcorruptlike/JpegCompression_primitive.py)                                |
| MotionBlur                         | [autovideo/augmentation/imgcorruptlike/MotionBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/MotionBlur_primitive.py)                                          |
| MotionBlur                         | [autovideo/augmentation/imgcorruptlike/MotionBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/MotionBlur_primitive.py)                                          |
| Pixelate                           | [autovideo/augmentation/imgcorruptlike/Pixelate_primitive.py](../autovideo/augmentation/imgcorruptlike/Pixelate_primitive.py)                                              |
| Saturate                           | [autovideo/augmentation/imgcorruptlike/Saturate_primitive.py](../autovideo/augmentation/imgcorruptlike/Saturate_primitive.py)                                              |
| ShotNoise                          | [autovideo/augmentation/imgcorruptlike/ShotNoise_primitive.py](../autovideo/augmentation/imgcorruptlike/ShotNoise_primitive.py)                                            |
| Snow                               | [autovideo/augmentation/imgcorruptlike/Snow_primitive.py](../autovideo/augmentation/imgcorruptlike/ShotNoise_primitive.py)                                                 |
| Spatter                            | [autovideo/augmentation/imgcorruptlike/Spatter_primitive.py](../autovideo/augmentation/imgcorruptlike/Spatter_primitive.py)                                                |
| SpeckleNoise                       | [autovideo/augmentation/imgcorruptlike/SpeckleNoise_primitive.py](../autovideo/augmentation/imgcorruptlike/SpeckleNoise_primitive.py)                                      |
| ZoomBlur                           | [autovideo/augmentation/imgcorruptlike/ZoomBlur_primitive.py](../autovideo/augmentation/imgcorruptlike/ZoomBlur_primitive.py)                                              |

## Meta
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ChannelShuffle                     | [autovideo/augmentation/meta/ChannelShuffle_primitive.py](../autovideo/augmentation/meta/ChannelShuffle_primitive.py)                                                      |
| ClipCBAsToImagePlanes              | [autovideo/augmentation/meta/ClipCBAsToImagePlanes_primitive.py](../autovideo/augmentation/meta/ClipCBAsToImagePlanes_primitive.py)                                        |
| Identity                           | [autovideo/augmentation/meta/Identity_primitive.py](../autovideo/augmentation/meta/Identity_primitive.py)                                                                  |
| RemoveCBAsByOutOfImageFraction     | [autovideo/augmentation/meta/RemoveCBAsByOutOfImageFraction_primitive.py](../autovideo/augmentation/meta/RemoveCBAsByOutOfImageFraction_primitive.py)                      |

## Pillike
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Autocontrast                       | [autovideo/augmentation/pillike/Autocontrast_primitive.py](../autovideo/augmentation/pillike/Autocontrast_primitive.py)                                                    |
| EnhanceBrightness                  | [autovideo/augmentation/pillike/EnhanceBrightness_primitive.py](../autovideo/augmentation/pillike/EnhanceBrightness_primitive.py)                                          |
| EnhanceColor                       | [autovideo/augmentation/pillike/EnhanceColor_primitive.py](../autovideo/augmentation/pillike/EnhanceColor_primitive.py)                                                    |
| EnhanceContrast                    | [autovideo/augmentation/pillike/EnhanceContrast_primitive.py](../autovideo/augmentation/pillike/EnhanceContrast_primitive.py)                                              |
| EnhanceSharpness                   | [autovideo/augmentation/pillike/EnhanceSharpness_primitive.py](../autovideo/augmentation/pillike/EnhanceSharpness_primitive.py)                                            |
| Equalize                           | [autovideo/augmentation/pillike/Equalize_primitive.py](../autovideo/augmentation/pillike/Equalize_primitive.py)                                                            |
| FilterBlur                         | [autovideo/augmentation/pillike/FilterBlur_primitive.py](../autovideo/augmentation/pillike/FilterBlur_primitive.py)                                                        |
| FilterContour                      | [autovideo/augmentation/pillike/FilterContour_primitive.py](../autovideo/augmentation/pillike/FilterContour_primitive.py)                                                  |
| FilterDetail                       | [autovideo/augmentation/pillike/FilterDetail_primitive.py](../autovideo/augmentation/pillike/FilterDetail_primitive.py)                                                    |
| FilterEdgeEnhanceMore              | [autovideo/augmentation/pillike/FilterEdgeEnhanceMore_primitive.py](../autovideo/augmentation/pillike/FilterEdgeEnhanceMore_primitive.py)                                  |
| FilterEdgeEnhance                  | [autovideo/augmentation/pillike/FilterEdgeEnhance_primitive.py](../autovideo/augmentation/pillike/FilterEdgeEnhance_primitive.py)                                          |
| FilterEmboss                       | [autovideo/augmentation/pillike/FilterEmboss_primitive.py](../autovideo/augmentation/pillike/FilterEmboss_primitive.py)                                                    |
| FilterFindEdges                    | [autovideo/augmentation/pillike/FilterFindEdges_primitive.py](../autovideo/augmentation/pillike/FilterFindEdges_primitive.py)                                              |
| FilterFindEdges                    | [autovideo/augmentation/pillike/FilterFindEdges_primitive.py](../autovideo/augmentation/pillike/FilterFindEdges_primitive.py)                                              |
| FilterSharpen                      | [autovideo/augmentation/pillike/FilterSharpen_primitive.py](../autovideo/augmentation/pillike/FilterSharpen_primitive.py)                                                  |
| FilterSmoothMore                   | [autovideo/augmentation/pillike/FilterSmoothMore_primitive.py](../autovideo/augmentation/pillike/FilterSmoothMore_primitive.py)                                            |
| FilterSmooth                       | [autovideo/augmentation/pillike/FilterSmooth_primitive.py](../autovideo/augmentation/pillike/FilterSmooth_primitive.py)                                                    |
| Solarize                           | [autovideo/augmentation/pillike/Solarize_primitive.py](../autovideo/augmentation/pillike/Solarize_primitive.py)                                                            |

## Pooling
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| AveragePooling                     | [autovideo/augmentation/pooling/AveragePooling_primitive.py](../autovideo/augmentation/pooling/AveragePooling_primitive.py)                                                |
| MaxPooling                         | [autovideo/augmentation/pooling/MaxPooling_primitive.py](../autovideo/augmentation/pooling/MaxPooling_primitive.py)                                                        |
| MedianPooling                      | [autovideo/augmentation/pooling/MedianPooling_primitive.py](../autovideo/augmentation/pooling/MedianPooling_primitive.py)                                                  |
| MinPooling                         | [autovideo/augmentation/pooling/MinPooling_primitive.py](../autovideo/augmentation/pooling/MinPooling_primitive.py)                                                        |

## Segmentation
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RegularGridVoronoi                 | [autovideo/augmentation/segmentation/RegularGridVoronoi_primitive.py](../autovideo/augmentation/segmentation/RegularGridVoronoi_primitive.py)                              |
| RelativeRegularGridVoronoi         | [autovideo/augmentation/segmentation/RelativeRegularGridVoronoi_primitive.py](../autovideo/augmentation/segmentation/RelativeRegularGridVoronoi_primitive.py)              |
| Superpixels                        | [autovideo/augmentation/segmentation/Superpixels_primitive.py](../autovideo/augmentation/segmentation/Superpixels_primitive.py)                                            |
| UniformVoronoi                     | [autovideo/augmentation/segmentation/UniformVoronoi_primitive.py](../autovideo/augmentation/segmentation/UniformVoronoi_primitive.py)                                      |

## Size
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| CenterCropToAspectRatio            | [autovideo/augmentation/size/CenterCropToAspectRatio_primitive.py](../autovideo/augmentation/size/CenterCropToAspectRatio_primitive.py)                                    |
| CenterCropToFixedSize              | [autovideo/augmentation/size/CenterCropToFixedSize_primitive.py](../autovideo/augmentation/size/CenterCropToFixedSize_primitive.py)                                        |
| CenterCropToMultiplesOf            | [autovideo/augmentation/size/CenterCropToMultiplesOf_primitive.py](../autovideo/augmentation/size/CenterCropToMultiplesOf_primitive.py)                                    |
| CenterCropToPowersOf               | [autovideo/augmentation/size/CenterCropToPowersOf_primitive.py](../autovideo/augmentation/size/CenterCropToPowersOf_primitive.py)                                          |
| CenterCropToSquare                 | [autovideo/augmentation/size/CenterCropToSquare_primitive.py](../autovideo/augmentation/size/CenterCropToSquare_primitive.py)                                              |
| CenterPadToAspectRatio             | [autovideo/augmentation/size/CenterPadToAspectRatio_primitive.py](../autovideo/augmentation/size/CenterPadToAspectRatio_primitive.py)                                      |
| CenterPadToFixedSize               | [autovideo/augmentation/size/CenterPadToFixedSize_primitive.py](../autovideo/augmentation/size/CenterPadToFixedSize_primitive.py)                                          |
| CenterPadToMultiplesOf             | [autovideo/augmentation/size/CenterPadToMultiplesOf_primitive.py](../autovideo/augmentation/size/CenterPadToMultiplesOf_primitive.py)                                      |
| CenterPadToPowersOf                | [autovideo/augmentation/size/CenterPadToPowersOf_primitive.py](../autovideo/augmentation/size/CenterPadToPowersOf_primitive.py)                                            |
| CenterPadToSquare                  | [autovideo/augmentation/size/CenterPadToSquare_primitive.py](../autovideo/augmentation/size/CenterPadToSquare_primitive.py)                                                |
| CropAndPad                         | [autovideo/augmentation/size/CropAndPad_primitive.py](../autovideo/augmentation/size/CropAndPad_primitive.py)                                                              |
| CropToAspectRatio                  | [autovideo/augmentation/size/CropToAspectRatio_primitive.py](../autovideo/augmentation/size/CropToAspectRatio_primitive.py)                                                |
| CropToFixSize                      | [autovideo/augmentation/size/CropToFixSize_primitive.py](../autovideo/augmentation/size/CropToFixSize_primitive.py)                                                        |
| CropToMultiplesOf                  | [autovideo/augmentation/size/CropToMultiplesOf_primitive.py](../autovideo/augmentation/size/CropToMultiplesOf_primitive.py)                                                |
| CropToPowersOf                     | [autovideo/augmentation/size/CropToPowersOf_primitive.py](../autovideo/augmentation/size/CropToPowersOf_primitive.py)                                                      |
| CropToSquare                       | [autovideo/augmentation/size/CropToSquare_primitive.py](../autovideo/augmentation/size/CropToSquare_primitive.py)                                                          |
| PadToAspectRatio                   | [autovideo/augmentation/size/PadToAspectRatio_primitive.py](../autovideo/augmentation/size/PadToAspectRatio_primitive.py)                                                  |
| PadToFixedSize                     | [autovideo/augmentation/size/PadToFixedSize_primitive.py](../autovideo/augmentation/size/PadToFixedSize_primitive.py)                                                      |
| PadToMultiplesOf                   | [autovideo/augmentation/size/PadToMultiplesOf_primitive.py](../autovideo/augmentation/size/PadToMultiplesOf_primitive.py)                                                  |
| PadToPowersOf                      | [autovideo/augmentation/size/PadToPowersOf_primitive.py](../autovideo/augmentation/size/PadToPowersOf_primitive.py)                                                        |
| PadToSquare                        | [autovideo/augmentation/size/PadToSquare_primitive.py](../autovideo/augmentation/size/PadToSquare_primitive.py)                                                            |
| Resize                             | [autovideo/augmentation/size/Resize_primitive.py](../autovideo/augmentation/size/Resize_primitive.py)                                                                      |

## Weather
| Augmentation Method                | Primitive Path                                                                                                                                                             |
| :--------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Clouds                             | [autovideo/augmentation/weather/Clouds_primitive.py](../autovideo/augmentation/weather/Clouds_primitive.py)                                                                |
| FastSnowyLandscape                 | [autovideo/augmentation/weather/FastSnowyLandscape_primitive.py](../autovideo/augmentation/weather/FastSnowyLandscape_primitive.py)                                        |
| Fog                                | [autovideo/augmentation/weather/Fog_primitive.py](../autovideo/augmentation/weather/Fog_primitive.py)                                                                      |
| Rain                               | [autovideo/augmentation/weather/Rain_primitive.py](../autovideo/augmentation/weather/Rain_primitive.py)                                                                    |
| Snowflakes                         | [autovideo/augmentation/weather/Snowflakes_primitive.py](../autovideo/augmentation/weather/Snowflakes_primitive.py)                                                        |

