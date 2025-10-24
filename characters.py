from abc import ABC, abstractmethod
from typing import List, Tuple

class Character(ABC):
    def __init__(self, radius: int|float, max_step: int|float):
        self._radius = radius
        self._max_step = max_step

    def get_radius(self) -> int|float: return self._radius
    def get_max_step(self) -> int|float: return self._max_step

    
class Player(Character): 
    def __init__(self, radius: int|float, max_step: int|float, start_pos: Tuple[int|float, int|float]):
        super().__init__(radius, max_step)
        self._start_pos = start_pos

    def get_start_pos(self) -> Tuple[float, float]: return self._start_pos

class Guard(Character):
    def __init__(self, radius: int|float, max_step: int|float, path: List[Tuple[int|float, int|float]]):
        super().__init__(radius, max_step)
        self._path = list(path)

    def get_path(self) -> List[Tuple[int|float, int|float]]:
        return list(self._path)