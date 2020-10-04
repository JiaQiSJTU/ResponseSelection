#!/usr/bin/env python3

# hack to make sure calling '-m transformer/ranker' works.
from .transformer import TransformerRankerAgent as RankerAgent  # noqa: F401
