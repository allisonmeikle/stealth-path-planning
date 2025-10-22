from __future__ import annotations

import math
import visilibity as vis

from typing import List, Optional, Tuple, Union
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import Geometry, LineString, MultiPolygon, Point, Polygon
from extremitypathfinder import PolygonEnvironment

from characters import *

class Map:
    _grid_size: Tuple[float, float]
    _boundary: List[Tuple[float, float]]
    _shapely_boundary: Polygon
    _obstacles: List[List[Tuple[float, float]]]
    _shapely_obstacles: List[Polygon]
    _visibility_polygons: List[List[Polygon]]
    _shadows: List[List[BaseGeometry]]
    _kernels: Optional[List[List[Kernel]]]

    def __init__(self, grid_size: Tuple[float, float], boundary: List[Tuple[float, float]], obstacles: List[List[Tuple[float, float]]], guards: List[Guard], player: Player):
        # Validate grid size
        if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(f"Invalid grid size {grid_size}: must be positive (width, height)")
        self._grid_size = grid_size

        # Validate boundary orientation (must be CCW)
        boundary_poly = Polygon(boundary)
        if not boundary_poly.is_valid:
            raise ValueError(f"Boundary polygon invalid: {explain_validity(boundary_poly)}")
        if not boundary_poly.exterior.is_ccw:
            raise ValueError("Boundary must be defined in counter-clockwise (CCW) order")
        self._boundary = boundary
        self._shapely_boundary = boundary_poly

        # Validate obstacles (must each be CW)
        self._obstacles = []
        self._shapely_obstacles = []
        for i, obs_coords in enumerate(obstacles):
            poly = Polygon(obs_coords)
            if not poly.is_valid:
                raise ValueError(f"Obstacle #{i} invalid: {explain_validity(poly)}")
            if poly.exterior.is_ccw:
                raise ValueError(f"Obstacle #{i} must be clockwise (CW), got CCW")
            self._obstacles.append(obs_coords)
            self._shapely_obstacles.append(poly)

        # --- Validate guards ---
        if not guards:
            raise ValueError("Map must contain at least one guard.")
        path_lengths = [len(g.get_path()) for g in guards]
        if len(set(path_lengths)) != 1:
            raise ValueError(
                f"All guards must have the same path length, but got lengths: {path_lengths}"
            )

        self._guards = guards
        self._num_timesteps = path_lengths[0]

        self._player = player

        # Build visibility environment (visilibity)
        self.build_visibility_environment()

        # Build pathfinding environment (extremitypathfinder)
        self.build_pathfinding_environment()
    
    def build_visibility_environment(self):
        try:
            outer = vis.Polygon([vis.Point(x, y) for x, y in self._boundary])
            holes = [vis.Polygon([vis.Point(x, y) for x, y in obstacle]) for obstacle in self._obstacles]
            visibility_env = vis.Environment([outer] + holes)
        except Exception as e:
            raise RuntimeError(f"Failed to build VisiLibity environment: {e}")

        if not visibility_env.is_valid():
            raise ValueError("Invalid VisiLibity environment built from map geometry")

        vis_polys = []
        shadows = []
        map_free = self._shapely_boundary.difference(unary_union(self._shapely_obstacles))

        for t in range(self._num_timesteps):
            step_vis_polys = []
            for guard in self._guards:
                guard_pos = guard.get_path()[t]
                try:
                    V = vis.Visibility_Polygon(vis.Point(*guard_pos), visibility_env, 1e-5)
                    coords = [(V[i].x(), V[i].y()) for i in range(V.n())]
                    poly = Polygon(coords).buffer(0)
                    step_vis_polys.append(poly)
                except Exception as e:
                    print(f"Warning: failed to compute visibility polygon at time {t} for guard at {guard_pos}: {e}")
                    continue  # skip this guard this timestep

            vis_polys.extend(step_vis_polys)
            shadow = map_free.difference(unary_union(step_vis_polys))
            shadows.append(shadow)

        self._visibility_polygons = vis_polys
        self._shadows = shadows

    def build_pathfinding_environment(self):
        try:
            boundary_poly = self._shapely_boundary
            obstacles_poly = self._shapely_obstacles
            if self._player.get_radius() > 0:
                # need to inflate the boundaries 
                boundary_poly = boundary_poly.buffer(-self._player.get_radius())
                obstacles_poly = [obstacle.buffer(self._player.get_radius()) for obstacle in obstacles_poly]
                
            boundary_coords = list(boundary_poly.exterior.coords)
            obstacle_coords = [list(obstacle.exterior.coords) for obstacle in obstacles_poly]

            self._polygon_env = PolygonEnvironment()
            self._polygon_env.store(boundary_coords, obstacle_coords, True)
            self._polygon_env.prepare()
        
        except Exception as e:
            raise RuntimeError(f"Failed to build pathfinding environment: {e}")
        
    def get_shortest_path(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        path, length = self._polygon_env.find_shortest_path(pt1, pt2)
        if (not path or not length):
            raise RuntimeError(f"Could not find path between {pt1} and {pt2}.")
        return path, length
    
    def compute_kernels(self, step): 
        self._kernels = []
        for i in range(len(self._shadows)):
            kernels = []
            for shadow in self._shadows[i]:
                kernels.extend(Map.Kernel.find_kernels(shadow, step, 0))
            self._kernels.extend(kernels)
        self._kernels = kernels

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
    
        @staticmethod
        def find_kernels(shape: BaseGeometry, step: float, depth : int) -> List[Map.Kernel]:
            kernels = []

            # Base cases: LineString, or Point
            if (shape.geom_type == 'LineString'):
                mid = shape.interpolate(0.5, normalized=True)  # midpoint
                if (isinstance(mid, Point)):
                    print(f"Got middle of a LineString at depth {depth}, returning point {mid}")
                    return [Map.Kernel((mid.x, mid.y), depth)]
            elif (shape.geom_type == 'Point'):
                print(f"Got a Point at depth {depth}, returning point {shape}")
                return [Map.Kernel((shape.x, shape.y), depth)]

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

        @staticmethod
        def get_kernel_point(poly: Polygon) -> Tuple[float, float]:
            centroid = poly.centroid
            if (poly.contains(centroid) and isinstance(centroid, Point)):
                return (centroid.x, centroid.y)
            rp = poly.representative_point()
            return (rp.x, rp.y)

        @staticmethod
        def get_moves_towards_kernels():
            moves = []
            if (self.depth == len(self.tree.shadows)-1):
                return moves
        
            for kernel in self.tree.kernels[self.depth+1]: 
                target = kernel.get_coords()
                path, length = self.tree.env.find_shortest_path(self.loc, target)
                if not path or length is None:
                    continue

                line = LineString(path)
                if length <= self.tree.max_step:
                    # Entire kernel is reachable
                    candidate =(line, target)
                else:
                    # Take a point along the path at max_step distance
                    pt = line.interpolate(self.tree.max_step)
                    # Build truncated path: from start → pt
                    truncated_coords = []
                    dist_so_far = 0.0
                    for i in range(len(path) - 1):
                        seg = LineString([path[i], path[i + 1]])
                        seg_len = seg.length
                        if dist_so_far + seg_len >= self.tree.max_step:
                            # Cut inside this segment
                            remaining = self.tree.max_step - dist_so_far
                            cut_pt = seg.interpolate(remaining)
                            truncated_coords.append((cut_pt.x, cut_pt.y))
                            break
                        else:
                            truncated_coords.append(path[i + 1])
                            dist_so_far += seg_len
                    truncated_line = LineString([path[0]] + truncated_coords)
                    candidate = (truncated_line, (pt.x, pt.y))
                    too_close = False
                    for _, existing_pt in moves:
                        if math.dist(existing_pt, candidate[1]) < prune_tol:
                            too_close = True
                            break
                    if not too_close:
                        moves.append(candidate)

            return moves