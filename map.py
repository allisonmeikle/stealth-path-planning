from __future__ import annotations
import math
import numpy as np
import os
import visilibity as vis

from typing import List, Optional, Tuple, Union
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import Geometry, LineString, MultiPolygon, Point, Polygon
from extremitypathfinder import PolygonEnvironment

from characters import *
from plot_helper import *

KERNEL_SHAPES: dict[int, list[tuple[BaseGeometry, int, list["Map.Kernel"]]]] = {}

class Map:
    _grid_size: Tuple[float, float]
    _boundary: List[Tuple[float, float]]
    _shapely_boundary: Polygon
    _obstacles: List[List[Tuple[float, float]]]
    _shapely_obstacles: List[Polygon]
    _visibility_polygons: List[MultiPolygon]
    _shadows: List[MultiPolygon]
    _shadows_w_obs: List[MultiPolygon]
    _kernels: Optional[List[List[Kernel]]]
    _kernels_w_obs: Optional[List[List[Kernel]]]
    _path_cache: dict

    def __init__(self, grid_size: Tuple[int|float, int|float], boundary: List[Tuple[int|float, int|float]], obstacles: List[List[Tuple[int|float, int|float]]], guards: List[Guard], player: Player):
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
        self._walkable_area = self._shapely_boundary.area
        for i, obs_coords in enumerate(obstacles):
            poly = Polygon(obs_coords)
            if not poly.is_valid:
                raise ValueError(f"Obstacle #{i} invalid: {explain_validity(poly)}")
            if poly.exterior.is_ccw:
                raise ValueError(f"Obstacle #{i} must be clockwise (CW), got CCW")
            self._obstacles.append(obs_coords)
            self._shapely_obstacles.append(poly)
            self._walkable_area -= poly.area

        # Pruning tolerance parameter α
        self._prune_alpha = 0.02 
        self._prune_tol = math.sqrt(self._walkable_area) * self._prune_alpha
        print(f"Map prune tolerance {self._prune_tol}")

        # --- Validate guards ---
        if not guards:
            raise ValueError("Map must contain at least one guard.")
        path_lengths = [len(g.get_path()) for g in guards]
        if len(set(path_lengths)) != 1:
            raise ValueError(
                f"All guards must have the same path length, but got lengths: {path_lengths}"
            )

        self._guards = guards
        self._player = player

        # Build visibility environment (visilibity)
        self.build_visibility_environment()

        # Build pathfinding environment (extremitypathfinder)
        self.build_pathfinding_environment()

        self._kernels = None
        self._path_cache = {}
    
    def build_visibility_environment(self):
        try:
            outer = vis.Polygon([vis.Point(x, y) for x, y in self._boundary])
            holes = []
            if self._obstacles:
                holes = [vis.Polygon([vis.Point(x, y) for x, y in obstacle]) for obstacle in self._obstacles]
            for hole in holes:
                print(hole)
            visibility_env = vis.Environment([outer] + holes)
        except Exception as e:
            raise RuntimeError(f"Failed to build VisiLibity environment: {e}")

        if not visibility_env.is_valid():
            raise ValueError("Invalid VisiLibity environment built from map geometry")

        vis_polys = []
        shadows = []
        shadows_w_obs = []
        map_free = Polygon(
            self._shapely_boundary.exterior.coords,
            holes=[obs.exterior.coords for obs in self._shapely_obstacles]
        )

        def sort_by_angle(coords, origin):
            ox, oy = origin
            return sorted(coords, key=lambda p: math.atan2(p[1]-oy, p[0]-ox))

        for t in range(self.get_num_timesteps()):
            polys = []
            for guard in self._guards:
                guard_pos = guard.get_path()[t]
                try:
                    V = vis.Visibility_Polygon(vis.Point(*guard_pos), visibility_env, 1e-5)
                    coords = [(V[i].x(), V[i].y()) for i in range(V.n())]
                    coords = sort_by_angle(coords, guard_pos)
                    poly = Polygon(coords).buffer(0)                    
                    poly = poly.intersection(map_free)
                    print(poly)
                    if poly.is_valid and not poly.is_empty:
                        polys.append(poly)
                except Exception as e:
                    print(f"Warning: failed to compute visibility polygon at time {t} for guard at {guard_pos}: {e}")
                    continue  # skip this guard this timestep
            
            for i, poly in enumerate(polys):
                if not poly.is_valid:
                    print(f"INVALID visibility poly at timestep {t}, index {i}")
                    print(explain_validity(poly))
                    print(poly.wkt)

            multi_poly = MultiPolygon(polys)
            vis_polys.append(multi_poly)

            map_free = Polygon(
                self._shapely_boundary.exterior.coords,
                holes=[obs.exterior.coords for obs in self._shapely_obstacles]
            )

            print("\n=== DEBUG: Checking validity before difference ===")
            print("map_free valid:", map_free.is_valid, explain_validity(map_free))
            print("multi_poly valid:", multi_poly.is_valid)
            for i, poly in enumerate(polys):
                if not poly.is_valid:
                    print(f"  visibility poly #{i} invalid:", explain_validity(poly))

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

        self._visibility_polygons = vis_polys
        self._shadows = shadows
        self._shadows_w_obs = shadows_w_obs

    def build_pathfinding_environment(self):
        try:
            boundary_poly = self._shapely_boundary
            obstacles_poly = self._shapely_obstacles

            if self._player.get_radius() > 0:
                # need to inflate the boundaries 
                boundary_poly = boundary_poly.buffer(-self._player.get_radius())
                obstacles_poly = [obstacle.buffer(self._player.get_radius()) for obstacle in obstacles_poly]
            
            boundary_poly = orient(boundary_poly, sign=1.0)   # CCW
            obstacles_poly = [orient(obs, sign=-1.0) for obs in obstacles_poly]  # CW

            boundary_coords = list(boundary_poly.exterior.coords)[:-1]
            obstacle_coords = [list(obstacle.exterior.coords)[:-1] for obstacle in obstacles_poly]

            self._polygon_env = PolygonEnvironment()
            self._polygon_env.store(boundary_coords, obstacle_coords, True)
            self._polygon_env.prepare()
        
        except Exception as e:
            raise RuntimeError(f"Failed to build pathfinding environment: {e}")

    def save_map_states(self, fig_size = (8,8), save_dir = "plots/map_states"):
        os.makedirs(save_dir, exist_ok=True)

        # Bounding Box
        minx, miny, maxx, maxy = self._shapely_boundary.bounds
        bounds = box(minx - 1, miny - 1, maxx + 1, maxy + 1)

        # Obstacles Union
        obstacles = unary_union(self._shapely_obstacles)
        
        for t in range(self.get_num_timesteps()):
            plt.figure(figsize=fig_size)
            ax = plt.gca()

            add_polygon(ax, bounds, fc="dimgray", ec="black", alpha=1.0, zorder=0)
            add_polygon(ax, self._shapely_boundary, fc="white", ec="black", alpha=1.0, zorder=1)
            add_polygon(ax, obstacles, fc="dimgray", ec="black", alpha=1.0, zorder=2)
            add_polygon(ax, self._shadows[t], fc="blue", alpha=0.3, zorder=3)

            for guard in self._guards:
                gx, gy = guard.get_path()[t]
                ax.plot(gx, gy, "r^", markersize=6, label="Guard")

            if self._kernels:
                for kernel in self.get_kernels(t):
                    kx, ky = kernel.get_coords()
                    ax.plot(kx, ky, "go", markersize=5, label="Kernel")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")
            ax.set_aspect("equal", adjustable="box")
            plt.xlim(minx - 1, maxx + 1)
            plt.ylim(miny - 1, maxy + 1)
            plt.title(f"Map State at Timestep {t}", fontsize=12)
            
            filename = os.path.join(save_dir, f"map_timestep_{t}.png")
            plt.savefig(filename, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved map state for timestep {t} → {filename}")

    def is_valid_position(self, pt: Tuple[int|float, int|float]):
        return self._polygon_env.within_map(np.asarray(pt))

    def is_same_position(self, p1: Tuple[int|float, int|float], p2: Tuple[int|float, int|float]):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx*dx + dy*dy) <= (self._prune_tol * self._prune_tol)
    
    def quantize_point(self, p):
        qx = round(p[0] / self._prune_tol)
        qy = round(p[1] / self._prune_tol)
        return (qx, qy)
    
    def get_player_start_pos(self):
        return self._player.get_start_pos(self)
    
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
        key = (round(pt1[0], 1), round(pt1[1], 1),round(pt2[0], 1), round(pt2[1], 1))
        if key in self._path_cache:
            #print("Path cache hit")
            return self._path_cache[key]
        path, length = self._polygon_env.find_shortest_path(pt1, pt2)
        if (not path or not length):
            raise RuntimeError(f"Could not find path between {pt1} and {pt2}.")
        self._path_cache[key] = (path, length)
        return path, length
    
    def get_visibility_polygon(self, timestep: int) -> MultiPolygon:
        if not(0 <= timestep < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve visibility polygon for invalid timetep {timestep}")
        return MultiPolygon([poly.buffer(0) for poly in self._visibility_polygons[timestep].geoms])
    
    def get_shadow(self, timestep: int) -> MultiPolygon:
        if not(0 <= timestep < self.get_num_timesteps()):
                raise IndexError(f"Could not retrieve shadow for invalid timetep {timestep}")
        return MultiPolygon([poly.buffer(0) for poly in self._shadows[timestep].geoms])
    
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

    def get_kernels(self, timestep: int):
        if not(0 <= timestep < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve kernels for invalid timetep {timestep}")
        
        if (not self._kernels or not self._kernels_w_obs):
            self._kernels = [[] for _ in range(self.get_num_timesteps())]
            self._kernels_w_obs = [[] for _ in range(self.get_num_timesteps())]

        if not self._kernels[timestep]:
            self._kernels[timestep] = self.find_kernels(self._shadows[timestep], 0.01, 0)
            self._kernels[timestep].sort(key=lambda k: k.get_depth(), reverse=True)
            self._kernels_w_obs[timestep] = self.find_kernels(self._shadows_w_obs[timestep], 0.01, 0)
            self._kernels_w_obs[timestep].sort(key=lambda k: k.get_depth(), reverse=True)
                
        return [Map.Kernel(k.get_coords(), k.get_depth()) for k in list(self._kernels[timestep] + self._kernels_w_obs[timestep])]
    
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