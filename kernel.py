import math
from typing import List, Tuple
from shapely import Geometry, Point, Polygon, LineString
from extremitypathfinder import PolygonEnvironment

class Kernel:
    def __init__(self, coords: Tuple[float, float], depth: int):
        self._coords = coords
        self._depth = depth
    
    def get_coords(self) -> Tuple[float, float]:
        return self._coords
    
    def get_depth(self) -> int:
        return self._depth
    
    def __str__(self) -> str:
        return f"Kernel (point=({self._coords[0]:.2f}, {self._coords[1]:.2f}), depth={self._depth})"


def find_kernels(shape: Geometry, step: float, depth : int) -> List[Kernel]:
    kernels = []

    # Base cases: LineString, or Point
    if (shape.geom_type == 'LineString'):
        mid = shape.interpolate(0.5, normalized=True)  # midpoint
        if (isinstance(mid, Point)):
            print(f"Got middle of a LineString at depth {depth}, returning point {mid}")
            return [Kernel((mid.x, mid.y), depth)]
    elif (shape.geom_type == 'Point'):
        print(f"Got a Point at depth {depth}, returning point {shape}")
        return [Kernel((shape.x, shape.y), depth)]

    # Recursive steps
    if (shape.geom_type == 'Polygon'):
        shrunk = shape.buffer(-step)
        if shrunk.is_empty:
            print(f"Polygon collapsed at depth {depth}, taking kernel point of last shape")
            return [Kernel(get_kernel_point(shape), depth)]
        kernels.extend(find_kernels(shrunk, step, depth + 1))
        return kernels
    
    # Composite type handling
    elif (shape.geom_type == 'MultiLineString'):
        for line in shape.geoms:
            kernels.extend(find_kernels(line, step, depth + 1))
        return kernels
    elif (shape.geom_type == 'MultiPolygon'):
        for poly in shape.geoms:
            kernels.extend(find_kernels(poly, step, depth + 1))
        return kernels
    
    print("Unsupported geometry type:", shape.geom_type)
    return []

def get_kernel_point(poly: Polygon) -> Tuple[float, float]:
    centroid = poly.centroid
    if (poly.contains(centroid) and isinstance(centroid, Point)):
        return (centroid.x, centroid.y)
    rp = poly.representative_point()
    return (rp.x, rp.y)

def get_reachable_kernels(env : PolygonEnvironment, max_step : float, loc : Tuple[float, float], kernels : List[Kernel]) -> List[Tuple[Kernel, LineString]]:
    reachable = []

    for k in kernels:
        target = k.get_coords()

        # Step 1: Euclidean shortcut check
        if math.dist(loc, target) > max_step:
            continue

        # Step 2: Compute shortest path
        path, length = env.find_shortest_path(loc, target)
        
        # No valid path found 
        if not length:
            continue

        # Convert to shapely LineString
        coords = [(p[0], p[1]) for p in path]
        line = LineString(coords)

        if length <= max_step:
            reachable.append((k, line))

    return reachable