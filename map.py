from __future__ import annotations
import imageio
import math
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
    _visibility_polygons: List[List[Polygon]]
    _shadows: List[MultiPolygon]
    _kernels: Optional[List[List[Kernel]]]

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
        self._player = player

        # Build visibility environment (visilibity)
        self.build_visibility_environment()

        # Build pathfinding environment (extremitypathfinder)
        self.build_pathfinding_environment()

        self._kernels = None
    
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

        for t in range(self.get_num_timesteps()):
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

            vis_polys.append(step_vis_polys)
            shadow = map_free.difference(unary_union(step_vis_polys))
            interior_free = self._shapely_boundary.buffer(-1e-3) 
            shadow = shadow.intersection(interior_free)
            if shadow.geom_type == "Polygon":
                shadow = MultiPolygon([shadow])
            elif shadow.geom_type == "GeometryCollection":
                shadow = MultiPolygon([g for g in shadow.geoms if g.geom_type == "Polygon"])
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

    @staticmethod
    def visualize_kernel_evolution(time_step: int, save_dir="plots/kernel_evolution", fps: int = 10):
        """
        Builds an animated GIF showing how the shadow polygons erode into their kernels
        over recursive depths recorded in KERNEL_SHAPES[time_step].
        Each frame draws all polygons at that stage and overlays all kernel points found so far.
        """
        from map import KERNEL_SHAPES  # import here to avoid circular deps
        os.makedirs(save_dir, exist_ok=True)

        if time_step not in KERNEL_SHAPES or not KERNEL_SHAPES[time_step]:
            print(f"⚠️ No recorded shapes for timestep {time_step}. Did you call find_kernels() with that timestep?")
            return

        frames = []
        print(f"🎞️ Building kernel evolution GIF for timestep {time_step} ({len(KERNEL_SHAPES[time_step])} stages)...")

        # Sort by recursion depth (so frames appear in chronological erosion order)
        shape_snapshots = sorted(KERNEL_SHAPES[time_step], key=lambda x: x[1])

        # Determine bounding box for consistent plotting
        all_bounds = [s.bounds for s, _, _ in shape_snapshots if hasattr(s, "bounds")]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # Build individual frames
        for idx, (shape, depth, kernels) in enumerate(shape_snapshots):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect("equal", "box")
            ax.set_title(f"Timestep {time_step} | Depth {depth} | {len(kernels)} kernels", fontsize=10)

            # Draw current geometry
            if shape.geom_type == "Polygon":
                x, y = shape.exterior.xy
                ax.fill(x, y, color="skyblue", alpha=0.4, ec="black")
                for hole in shape.interiors:
                    hx, hy = zip(*hole.coords)
                    ax.fill(hx, hy, color="white")
            elif shape.geom_type == "MultiPolygon":
                for poly in shape.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color="skyblue", alpha=0.4, ec="black")

            # Draw all kernel points found so far
            for k in kernels:
                kx, ky = k.get_coords()
                ax.plot(kx, ky, "ro", markersize=4)

            ax.set_xlim(minx - 0.5, maxx + 0.5)
            ax.set_ylim(miny - 0.5, maxy + 0.5)
            ax.axis("off")

            frame_path = os.path.join(save_dir, f"frame_t{time_step}_d{depth:03d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            frames.append(frame_path)

        # Combine into GIF
        images = [imageio.imread(f) for f in frames]
        output_gif = os.path.join(save_dir, f"kernel_evolution_t{time_step}.gif")
        imageio.mimsave(output_gif, images, duration=1/fps)
        print(f"✅ Saved GIF → {output_gif}")
    
    def get_player_start_pos(self):
        return self._player.get_start_pos()
    
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
        print(f"Finding path from {pt1} to {pt2}")
        path, length = self._polygon_env.find_shortest_path(pt1, pt2)
        if (not path or not length):
            raise RuntimeError(f"Could not find path between {pt1} and {pt2}.")
        return path, length
    
    def get_visibility_polygons(self):
        """Return a deep copy of the visibility polygons to prevent external mutation."""
        return [
            [poly.buffer(0) for poly in timestep]  # shapely .buffer(0) clones geometry
            for timestep in self._visibility_polygons
        ]
    
    def get_kernels(self, time_step: int):
        if self._kernels is None:
            self.compute_kernels(0.01)
        
        if not(0 <= time_step < self.get_num_timesteps()):
            raise IndexError(f"Could not retrieve kernels for invalid timetep {time_step}")
        
        return [Map.Kernel(k.get_coords(), k.get_depth()) for k in self._kernels[time_step]]

    def compute_kernels(self, step_factor: float): 
        self._kernels = []
        for i in range(len(self._shadows)):
            kernels = Map.Kernel.find_kernels(self._shadows[i], step_factor, 0, i, [])
            self._kernels.append(kernels)    

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
        def find_kernels(shape: BaseGeometry, step_factor: float, depth: int, time_step: int, cur_kernels: List[Map.Kernel]) -> List[Map.Kernel]:
            from map import KERNEL_SHAPES
            if time_step not in KERNEL_SHAPES:
                KERNEL_SHAPES[time_step] = []
            KERNEL_SHAPES[time_step].append((shape, depth, cur_kernels.copy()))

            # Base case(s)
            if shape.is_empty:
                print(f"Got an empty shape at depth {depth}")
                return cur_kernels
            elif (shape.geom_type == 'LineString'):
                start_pt = Point(shape.coords[0])
                end_pt = Point(shape.coords[-1])
                mid_pt = shape.interpolate(0.5, normalized=True)

                # Create kernels for each
                for pt in (start_pt, mid_pt, end_pt):
                    k = Map.Kernel((pt.x, pt.y), depth)
                    cur_kernels.append(k)                
                return cur_kernels

            # Tail-Recursive steps
            elif (shape.geom_type == 'MultiPolygon'):
                for subpoly in shape.geoms:
                    cur_kernels = Map.Kernel.find_kernels(subpoly, step_factor, depth + 1, time_step, cur_kernels)
                return cur_kernels
            elif (shape.geom_type == 'Polygon'):
                adaptive_step = max(step_factor * math.sqrt(shape.area), 0.01)
                shrunk = shape.buffer(-adaptive_step)

                # Handle base-cases here too before we lose the shape
                if shrunk.is_empty or shrunk.equals(shape):
                    k = Map.Kernel(Map.Kernel.get_kernel_point(shape), depth)
                    cur_kernels.append(k)
                    KERNEL_SHAPES[time_step].append((shape, depth, cur_kernels.copy()))

                return Map.Kernel.find_kernels(shrunk, step_factor, depth + 1, time_step, cur_kernels)
            else:
                print("Unsupported geometry type:", shape.geom_type)
                return []

        @staticmethod
        def get_kernel_point(poly: Polygon) -> Tuple[float, float]:
            centroid = poly.centroid
            if (poly.contains(centroid) and isinstance(centroid, Point)):
                return (centroid.x, centroid.y)
            rp = poly.representative_point()
            return (rp.x, rp.y)