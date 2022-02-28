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
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from d3m import container
from d3m.metadata import hyperparams
from d3m.primitive_interfaces.base import *
from d3m.primitive_interfaces import base, transformer
import abc
from autovideo.utils import construct_primitive_metadata
import imgaug.augmenters as iaa

Inputs = container.DataFrame
Outputs = container.DataFrame

__all__ = ("WithChannelsPrimitive")


class Hyperparams(hyperparams.Hyperparams):
    channels = hyperparams.Constant(
                default=None,
                description='Sets the channels to be extracted from each image. Default `None` use all channels. List of ints',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

class WithChannelsPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    metadata = construct_primitive_metadata("augmentation", "meta_WithChannels")

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        augmentors = list(filter(None, inputs["augmentation"].tolist()))
        print(self.hyperparams["channels"])
        function = iaa.WithChannels(channels=self.hyperparams["channels"], children=augmentors)
        aug = [function]
        aug.extend(["" for i in range(inputs.shape[0]-1)]) # only store in the first row
        inputs["augmentation"] = aug
        output = Outputs(inputs)
        return base.CallResult(output)
