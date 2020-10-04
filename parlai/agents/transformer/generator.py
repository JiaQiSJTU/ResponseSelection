#!/usr/bin/env python3


# hack to make sure -m transformer/generator works as expected
from .transformer import TransformerGeneratorAgent as GeneratorAgent  # noqa: F401
