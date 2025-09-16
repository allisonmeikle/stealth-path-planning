import visilibity as vis
import pyvisgraph as vg
from typing import List
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from kernel import *

def shapely_to_visilibity_polygon(poly: Polygon, epsilon=1e-5):
    if not poly.is_valid:
        raise ValueError("Invalid shapely polygon")

    poly = orient(poly, sign=1.0)
    exterior_coords = list(poly.exterior.coords)[:-1]
    outer = vis.Polygon([vis.Point(x, y) for x, y in exterior_coords])

    holes = []
    for hole in poly.interiors:
        hole_coords = list(hole.coords)[:-1]
        if Polygon(hole_coords).exterior.is_ccw:
            hole_coords = list(reversed(hole_coords))
        hpoly = vis.Polygon([vis.Point(x, y) for x, y in hole_coords])
        holes.append(hpoly)

    return outer, holes


def compute_visibility_polygon(map_poly, obstacles, guard_pos, epsilon=1e-5):
    guard = vis.Point(*guard_pos)

    map_poly = orient(map_poly, sign=1.0)
    outer_coords = list(map_poly.exterior.coords)[:-1]
    outer = vis.Polygon([vis.Point(x, y) for x, y in outer_coords])

    holes = []
    for obs in obstacles:
        hole_coords = list(obs.exterior.coords)[:-1]
        if Polygon(hole_coords).exterior.is_ccw:
            hole_coords = list(reversed(hole_coords))
        hpoly = vis.Polygon([vis.Point(x, y) for x, y in hole_coords])
        holes.append(hpoly)

    env = vis.Environment([outer] + holes)

    if not env.is_valid():
        raise ValueError("Invalid VisiLibity environment")

    V = vis.Visibility_Polygon(guard, env, epsilon)
    coords = [(V[i].x(), V[i].y()) for i in range(V.n())]
    poly = Polygon(coords).buffer(0)
    return poly

def compute_shadows(map : Polygon, obstacles : List[Polygon], guard_positions : List[Point]) -> List[BaseGeometry]: 
    shadows = []
    for i in range (len(guard_positions)):
        vis_poly = compute_visibility_polygon(map, obstacles, (guard_positions[i].x, guard_positions[i].y))
        map_free = map.difference(unary_union(obstacles))
        shadows.append(map_free.difference(vis_poly))
    return shadows

def compute_kernels(shadows : List[BaseGeometry], step : float) -> List[List[Kernel]]:
    kernels = []
    for i in range(len(shadows)):
        kernels.append(find_kernels(shadows[i], step, 0))
    return kernels

def build_visibility_graph(map_poly: Polygon, obstacles: list[Polygon]) -> vg.VisGraph:
    import pyvisgraph as vg

    # Convert shapely polygons into pyvisgraph polygons
    def shapely_to_vg(poly: Polygon):
        return [vg.Point(x, y) for x, y in poly.exterior.coords[:-1]]  # exclude closing point

    # Obstacles + map boundary walls
    polygons = [shapely_to_vg(obs) for obs in obstacles]

    # Add the map boundary as a "polygon obstacle"
    polygons.append(shapely_to_vg(map_poly))

    # Build the graph
    graph = vg.VisGraph()
    graph.build(polygons)
    return graph
