from __future__ import annotations
import math
import numpy as np
import os
import visilibity as vis

from typing import List, Optional, Tuple, Union
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, nearest_points
from shapely import Geometry, LineString, MultiPolygon, Point, Polygon
from extremitypathfinder import PolygonEnvironment

from characters import *

class Map:
    def __init__(self, level: str, grid_size: Tuple[int|float, int|float], boundary: List[Tuple[int|float, int|float]], obstacles: List[List[Tuple[int|float, int|float]]], guards: List[Guard], player: Player):
        self._level = level
        
        # Validate grid size
        if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError(f"Invalid grid size {grid_size}: must be positive (width, height)")
        self._grid_size = grid_size
        self._guards = guards
        self._player = player
        
        boundary_poly = orient(Polygon(boundary), sign=1.0) # CCW
        if not boundary_poly.is_valid:
            raise ValueError(f"Boundary polygon invalid: {explain_validity(boundary_poly)}")
        self._boundary = boundary
        self._shapely_boundary = boundary_poly    
        self._shapely_boundary_inflated = orient(boundary_poly.buffer(-self._player.get_radius()), sign=1.0) # CCW

        self._obstacles = []
        self._shapely_obstacles = []
        self._shapely_obstacles_inflated = []
        for i, obs_coords in enumerate(obstacles):
            self._obstacles.append(obs_coords)

            poly = Polygon(obs_coords)
            poly = orient(poly, sign=-1.0) # CW
            if not poly.is_valid:
                raise ValueError(f"Obstacle #{i} invalid: {explain_validity(poly)}")
            self._shapely_obstacles.append(poly)
            self._shapely_obstacles_inflated.append(orient(poly.buffer(self._player.get_radius()), sign=-1.0))

        # Calculate walkable area
        self._shapely_walkable_area = self._shapely_boundary_inflated.difference(unary_union(self._shapely_obstacles_inflated))
        # Pruning tolerance parameter α
        self._prune_alpha = 0.02
        self._prune_tol = math.sqrt(self._shapely_walkable_area.area) * self._prune_alpha
        print(f"Got prune tolerance {self._prune_tol}")
        self._min_dist_btw_kernels = self._prune_tol * 10.0

        # Build visibility environment (visilibity)
        self.build_visibility_environment()

        # Build pathfinding environment (extremitypathfinder)
        self.build_pathfinding_environment()

        self._kernels = None
        self._kernels_w_obs = None
        self._kernels_merged = None
        self._kernels_diverse = None
        self._path_cache = {}
        self._path_cache_queries = 0
        self._path_cache_hits = 0

    @staticmethod
    def extract_polygons(shape) -> list[Polygon]:
        polys = []
        if isinstance(shape, Polygon):
            polys.append(shape)
        elif isinstance(shape, MultiPolygon):
            polys.extend(list(shape.geoms))
        elif shape.geom_type == "GeometryCollection":
            polys.extend([p for p in shape.geoms if isinstance(p, Polygon)])

        valid_polys = []
        for poly in polys:
            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                valid_polys.append(poly)
        return valid_polys

    def build_visibility_environment(self):
        try:
            outer = vis.Polygon([vis.Point(x, y) for x, y in self._boundary])
            holes = []
            if self._obstacles:
                holes = [vis.Polygon([vis.Point(x, y) for x, y in obstacle]) for obstacle in self._obstacles]
            visibility_env = vis.Environment([outer] + holes)
        except Exception as e:
            raise RuntimeError(f"Failed to build VisiLibity environment: {e}")

        if not visibility_env.is_valid():
            raise ValueError("Invalid VisiLibity environment built from map geometry")

        vis_polys = []
        vis_polys_raw = []
        shadows = []
        shadows_w_obs = []
        map_free = Polygon(
            self._shapely_boundary.exterior.coords,
            holes=[obs.exterior.coords for obs in self._shapely_obstacles]
        )

        for t in range(self.get_num_timesteps()):
            polys_raw = []
            polys = []
            for guard in self._guards:
                guard_pos = guard.get_path()[t]
                try:
                    V = vis.Visibility_Polygon(vis.Point(*guard_pos), visibility_env, 1e-5)
                    polys_raw.append(V)
                    coords = [(V[i].x(), V[i].y()) for i in range(V.n())]
                    poly = Polygon(coords).buffer(0)
                    polys.extend(Map.extract_polygons(poly))
                except Exception as e:
                    print(f"Warning: failed to compute visibility polygon at time {t} for guard at {guard_pos}: {e}")
                    continue  # skip this guard this timestep
            
            multi_poly = MultiPolygon(polys).buffer(0)
            multi_poly = MultiPolygon(Map.extract_polygons(multi_poly))
            vis_polys.append(multi_poly)
            vis_polys_raw.append(polys_raw)

            shadow = map_free.difference(multi_poly)
            shadow = shadow.difference(unary_union(self._shapely_obstacles)) 
            interior_free = self._shapely_boundary.buffer(-1e-3) 
            shadow = shadow.intersection(interior_free)

            # Ensure interior holes (obstacles) are correctly oriented (CW)
            if shadow.geom_type == "Polygon":
                shadow = orient(shadow, sign=1.0)
                shadow = MultiPolygon([shadow])
            elif shadow.geom_type == "MultiPolygon":
                shadow = MultiPolygon([orient(s, sign=1.0) for s in shadow.geoms])
            elif shadow.geom_type == "GeometryCollection":
                shadow = MultiPolygon([orient(g, sign=1.0) for g in shadow.geoms if g.geom_type == "Polygon"])
            shadows.append(shadow)

            # Add interior obstacles touching the shadow back into the shadow
            for obs in self._shapely_obstacles:
                if shadow.intersects(obs):
                    shadow = shadow.union(obs)
            shadows_w_obs.append(shadow)

        self._visibility_polygons = vis_polys_raw
        self._shapely_visibility_polygons = vis_polys
        self._shadows = shadows
        self._shadows_w_obs = shadows_w_obs

    def build_pathfinding_environment(self):
        try:
            boundary_coords = list(self._shapely_boundary_inflated.exterior.coords)[:-1]
            obstacle_coords = [list(obstacle.exterior.coords)[:-1] for obstacle in self._shapely_obstacles_inflated]

            self._polygon_env = PolygonEnvironment()
            self._polygon_env.store(boundary_coords, obstacle_coords, True)
        
        except Exception as e:
            raise RuntimeError(f"Failed to build pathfinding environment: {e}")
    
    def get_closest_valid_pt(self, pt: Tuple[int|float, int|float]) -> Tuple[int|float, int|float]:
        if self.is_valid_position(pt):
            return pt
        p = Point(pt)
        nearest_geom, _ = nearest_points(self._shapely_walkable_area, p)
        return (nearest_geom.x, nearest_geom.y)

    def is_valid_position(self, pt: Tuple[int|float, int|float]):
        return self._polygon_env.within_map(np.asarray(pt))

    def is_same_position(self, p1: Tuple[int|float, int|float], p2: Tuple[int|float, int|float]):
        return math.dist(p1, p2) <= (self._prune_tol)
    
    def quantize_point(self, p):
        qx = round(p[0] / self._prune_tol)
        qy = round(p[1] / self._prune_tol)
        return (qx, qy)
    
    def is_visible(self, p: Tuple[int|float, int|float], timestep: int):
        if not(0 <= timestep < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve visibility polygon for invalid timetep {timestep}")
        for vis_poly in self._visibility_polygons[timestep]:
            coords = [(vis_poly[i].x(), vis_poly[i].y()) for i in range(vis_poly.n())]
            poly = Polygon(coords)
            if poly.covers(Point(p)):
                #print(f"{p} is visible at timestep {timestep}")
                return True
        return False
    
    def get_player_start_pos(self):
        x, y = self._player.get_start_pos(self)
        return (round(x, 2), round(y, 2))
    
    def get_player_max_step(self):
        return self._player.get_max_step()
    
    def get_guard_positions(self, time_step: int):
        positions = []
        for guard in self._guards:
            if not(0 <= time_step < len(guard.get_path())):
                raise IndexError(f"Could not retrieve guard positions for invalid timetep {time_step}")
            positions.append(guard.get_path()[time_step])
        return positions
    
    def get_num_timesteps(self):
        return len(self._guards[0].get_path())
        
    def get_shortest_path(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        self._path_cache_queries += 1
        key = (self.quantize_point(pt1), self.quantize_point(pt2))
        if key in self._path_cache:
            self._path_cache_hits += 1
            return self._path_cache[key]
        
        if not self.is_valid_position(pt1):
            pt1 = self.get_closest_valid_pt(pt1)
        if not self.is_valid_position(pt2):
            pt2 = self.get_closest_valid_pt(pt2)

        try:
            path, length = self._polygon_env.find_shortest_path(pt1, pt2)
        except Exception as e:
            print(f"Got exception in path planning between {pt1} and {pt2}. {e}")
            return ([], 0.0)
        
        if (len(path) == 0 or length is None):
            print(f"Could not find shortest path  between {pt1} and {pt2}.")
            return ([], 0.0)
        self._path_cache[key] = (path, length)
        return path, length
    
    def get_longest_move_along_shortest_path(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Optional[Tuple[List[Tuple[float, float]], Tuple[int|float, int|float]]]:
        path, length = self.get_shortest_path(pt1, pt2)
        max_step = self._player.get_max_step()
        
        # Pt2 is reachable
        if length <= max_step:
            return (path, pt2)
        # Path needs truncation
        truncated_path = [path[0]]
        dist_left = max_step

        for a, b in zip(path, path[1:]):
            ax, ay = a
            bx, by = b

            dx = bx - ax
            dy = by - ay
            seg_len = math.hypot(dx, dy)
            #print(f"Got pt on truncated path: {bx, by}")

            if seg_len < 1e-12:
                # skip degenerate segment
                continue
            
            # If the truncated point lies inside this segment
            if seg_len >= dist_left:
                ratio = dist_left / seg_len
                new_x = ax + dx * ratio
                new_y = ay + dy * ratio
                new_pt = (new_x, new_y)

                # Validate / snap to valid point
                if not self.is_valid_position(new_pt):
                    new_pt = (ax, ay)

                truncated_path.append(new_pt)
                return (truncated_path, new_pt)

            # Else: consume this entire segment
            truncated_path.append(b)
            dist_left -= seg_len

        # If we reach here something failed unexpectedly
        print(f"Failed to truncate path from {pt1} to {pt2}")
        return None
    
    def get_visibility_polygon(self, timestep: int) -> MultiPolygon:
        if not(0 <= timestep < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve visibility polygon for invalid timetep {timestep}")
        return self._shapely_visibility_polygons[timestep]
    
    def get_shadow(self, timestep: int) -> MultiPolygon:
        if not(0 <= timestep < self.get_num_timesteps()):
                raise IndexError(f"Could not retrieve shadow for invalid timetep {timestep}")
        return self._shadows[timestep]

    def find_kernels(self, shape: BaseGeometry, step_factor: float, depth: int) -> List[Map.Kernel]: 
        kernels = []
        # Base case(s)
        if shape.is_empty:
            return kernels
        elif (shape.geom_type == 'LineString'):
            mid_pt = shape.interpolate(0.5, normalized=True)
            for pt in (shape.coords[0], (mid_pt.x, mid_pt.y), shape.coords[-1]):
                if self.is_valid_position(pt):
                    kernels.append(Map.Kernel(pt, depth))                
                return kernels

        # Recursive steps
        elif (shape.geom_type == 'MultiPolygon'):
            for subpoly in shape.geoms:
                kernels.extend(self.find_kernels(subpoly, step_factor, depth + 1))
        elif (shape.geom_type == 'Polygon'):
            adaptive_step = max(step_factor * math.sqrt(shape.area), 0.01)
            shrunk = shape.buffer(-adaptive_step)

            # Handle base-cases here too before we lose the shape
            if shrunk.is_empty or shrunk.equals(shape):
                centroid = shape.centroid
                pt = (centroid.x, centroid.y)
                if not shape.contains(centroid):
                    rp = shape.representative_point()
                    pt = (rp.x, rp.y)
                
                if (self.is_valid_position(pt)):
                    kernels.append(Map.Kernel(pt, depth))
                return kernels

            return self.find_kernels(shrunk, step_factor, depth + 1)
        else:
            print("Unsupported geometry type:", shape.geom_type)
        
        return kernels

    def get_kernels(self, timestep: int, diverse=False) -> List[Kernel]:
        if not(0 <= timestep < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve kernels for invalid timetep {timestep}")
        
        if (self._kernels is None or self._kernels_w_obs is None or self._kernels_merged is None or self._kernels_diverse is None):
            self._kernels = [[] for _ in range(self.get_num_timesteps())]
            self._kernels_w_obs = [[] for _ in range(self.get_num_timesteps())]
            self._kernels_merged = [[] for _ in range(self.get_num_timesteps())]
            self._kernels_diverse = [[] for _ in range(self.get_num_timesteps())]

        if not self._kernels[timestep] or not self._kernels_w_obs[timestep] or not self._kernels_merged[timestep]:
            self._kernels[timestep] = self.find_kernels(self._shadows[timestep], 0.01, 0)
            self._kernels[timestep].sort(key=lambda k: k.get_depth(), reverse=True)
            self._kernels_w_obs[timestep] = self.find_kernels(self._shadows_w_obs[timestep], 0.01, 0)
            self._kernels_w_obs[timestep].sort(key=lambda k: k.get_depth(), reverse=True)

            self._kernels_merged[timestep] = list(self._kernels[timestep] + self._kernels_w_obs[timestep])
            self._kernels_merged[timestep].sort(key=lambda k: k.get_depth(), reverse=True)

            if diverse:
                if not self._kernels_diverse[timestep]:
                    for kernel in self._kernels_merged[timestep]:
                        x, y = kernel.get_coords()
                        #print(f"checking kernel coords {x, y}")
                        # distance check against previously taken kernels
                        too_close = any(
                            (x - ker.get_coords()[0])**2 + (y - ker.get_coords()[1])**2 < self._min_dist_btw_kernels**2
                            for ker in self._kernels_diverse[timestep]
                        )
                        if not too_close:
                            #print(f"adding coords {x, y}")
                            self._kernels_diverse[timestep].append(kernel)   
                #print(f"Got {len(self._kernels_diverse[timestep])} diverse kernels from {len(self._kernels_merged[timestep])}")
                return self._kernels_diverse[timestep]

        return self._kernels_merged[timestep]
    
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