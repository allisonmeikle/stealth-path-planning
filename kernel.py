import pyvisgraph as vg

from typing import List, Tuple
from shapely import Geometry, Point, Polygon, LineString

class Kernel:
    def __init__(self, point: Point, depth: int):
        self._point = point
        self._depth = depth
    
    def get_point(self) -> Point:
        return self._point
    
    def get_depth(self) -> int:
        return self._depth
    
    def __str__(self) -> str:
        return f"Kernel (point=({self._point.x:.2f}, {self._point.y:.2f}), depth={self._depth})"


def find_kernels(shape: Geometry, step: float, depth : int) -> List[Kernel]:
    kernels = []

    # Base cases: LineString, or Point
    if (shape.geom_type == 'LineString'):
        mid = shape.interpolate(0.5, normalized=True)  # midpoint
        if (isinstance(mid, Point)):
            print(f"Got middle of a LineString at depth {depth}, returning point {mid}")
            return [Kernel(mid, depth)]
    elif (shape.geom_type == 'Point'):
        print(f"Got a Point at depth {depth}, returning point {shape}")
        return [Kernel(shape, depth)]

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

def get_kernel_point(poly: Polygon) -> Point:
    centroid = poly.centroid
    if (poly.contains(centroid) and isinstance(centroid, Point)):
        return centroid
    return poly.representative_point()

def get_reachable_kernels(graph : vg.VisGraph, max_step : float, loc : Point, kernels : List[Kernel]) -> List[Tuple[Kernel, LineString]]:
    reachable = []
    start = vg.Point(loc.x, loc.y)

    for k in kernels:
        kp = k.get_point()
        target_pt = vg.Point(kp.x, kp.y)

        # Step 1: Euclidean shortcut check
        if loc.distance(kp) > max_step:
            continue

        # Step 2: Compute shortest path
        path = graph.shortest_path(start, target_pt)
        if not path:
            continue

        # Convert to shapely LineString
        coords = [(p.x, p.y) for p in path]
        line = LineString(coords)

        if line.length <= max_step:
            reachable.append((k, line))

    return reachable