#!/usr/bin/env python3


"""
Definitions of general ParlAI types.
"""
from typing import Any, Dict, TypeVar, Union

import torch


_Scalar = Union[int, float, torch.Tensor]
"""
ParlAI type to represent an object that is theoretically expressible as a scalar value.
Ints and floats are clearly scalars, and torch.Tensors can be represented by a scalar if
Tensor.numel() == 1. Used as input type for classes derived from Metric.

Note that _Scalar cannot be defined as a subclass of Union, analogously to _Shared
below, because Union does not support subclassing.
"""


class _Shared(Dict[str, Any]):
    """
    ParlAI ``shared`` Structure.

    The `shared` dict that is used to instantiate shared agents in ParlAI,
    e.g. when using batching, distributed training, etc.

    Type is ``TShared``.
    """


TShared = TypeVar('TShared', bound=_Shared)

TScalar = TypeVar('TScalar', bound=_Scalar)
