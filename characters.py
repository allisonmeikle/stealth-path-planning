from abc import ABC, abstractmethod
from typing import List, Tuple

class Character(ABC):
    def __init__(self, radius: float, max_step: float):
        self._radius = radius
        self._max_step = max_step

    def get_radius(self) -> float: return self._radius
    def get_max_step(self) -> float: return self._max_step

    
class Player(Character): 
    def __init__(self, radius: float, max_step: float, start_pos: Tuple[float, float]):
        super().__init__(radius, max_step)
        self._start_pos = start_pos

    def get_start_pos(self) -> Tuple[float, float]: return self._start_pos

class Guard(Character):
    def __init__(self, radius: float, max_step: float, path: List[Tuple[float, float]]):
        super().__init__(radius, max_step)
        self._path = list(path)

    def get_path(self) -> List[Tuple[float, float]]:
        return list(self._path)