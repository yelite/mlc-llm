from dataclasses import dataclass
from typing import Optional

from .api import APIConfig


@dataclass
class InferenceEngineConfig:
    max_num_batched_tokens: Optional[int]
    max_num_sequences: int


@dataclass
class ModelConfig:
    pass


@dataclass
class RootConfig:
    api: APIConfig
    inference_engine: InferenceEngineConfig
    model: ModelConfig
