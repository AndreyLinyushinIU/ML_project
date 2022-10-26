from dataclasses import dataclass
from abc import ABC
from typing import Optional, Generator

import numpy as np


@dataclass
class Model(ABC):
    id: int
    name: str
    description: str

    def run_and_save(self, content_image, style_image: np.ndarray, result_image_path: str):
        raise NotImplementedError()


class ModelsRegistry:

    def __init__(self):
        self._id = 1
        self._models = dict()

    def register(self, model: Model):
        model.id = str(self._id)
        self._models[str(self._id)] = model
        self._id += 1

    def __iter__(self) -> Generator[Model, None, None]:
        for model in self._models.values():
            yield model

    def get(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)
