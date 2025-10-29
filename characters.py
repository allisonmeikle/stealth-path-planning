from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from map import *

class Character(ABC):
    def __init__(self, radius: int|float, max_step: int|float):
        self._radius = radius
        self._max_step = max_step

    def get_radius(self) -> int|float: return self._radius
    def get_max_step(self) -> int|float: return self._max_step

    
class Player(Character): 
    def __init__(self, radius: int|float, max_step: int|float, start_pos: Optional[Tuple[int|float, int|float]]):
        super().__init__(radius, max_step)
        self._start_pos = start_pos

    def get_start_pos(self, map: Optional[Map] = None) -> Tuple[float, float]: 
        if not self._start_pos and map:
            shadow = map.get_shadow(0)
            if shadow.is_empty:
                raise RuntimeError("No shadow region is available at timestep 0.")
            
            polys = list(shadow.geoms)
            areas = [p.area for p in polys]
            total = sum(areas)
            weights = [a / total for a in areas]

            for _ in range(100):
                chosen = random.choices(polys, weights=weights, k=1)[0]
                buffered = chosen.buffer(-self.get_radius())
                if not buffered.is_empty:
                    minx, miny, maxx, maxy = buffered.bounds
                    p = (random.uniform(minx, maxx), random.uniform(miny, maxy))
                    if buffered.contains(Point(p[0], p[1])) and map.is_valid_position(p):
                        self._start_pos = p
                        return p
            raise RuntimeError(f"Could not generate a valid player starting position")
        return self._start_pos

class Guard(Character):
    def __init__(self, radius: int|float, max_step: int|float, path: List[Tuple[int|float, int|float]]):
        super().__init__(radius, max_step)
        self._path = list(path)

    def get_path(self) -> List[Tuple[int|float, int|float]]:
        return list(self._path)