from __future__ import annotations

import math
import os

from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString

from characters import *
from map import *
from plot_helper import *
from helpers import *

class MonteCarloTree: 
    c = math.sqrt(2)

    def __init__(self, map: Map, player: Player, guard: Guard):
        self._map = map
        self._player = player
        self._guard = guard

        self.root = MonteCarloTree.Node(self, 0, player.get_start_pos(), False)
        self.max_depth = len(guard.get_path()) - 1

        # Score hyperparameters
        self.alpha = 1 # for shadow score
        self.beta = 1 # for guard distance score
        self.gamma = -1 # for kernel distance score

    def get_max_step(self):
        return self._player.get_max_step()
    
    def get_shortest_path(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        return self._map.get_shortest_path(pt1, pt2)
        
    def select(self) -> Optional[MonteCarloTree.Node]:
        current = self.root
        while True:
            print("Current node in selection loop: ", current)

            if (current.depth >= self.max_depth):
                print("Selection ended on a leaf, returning None")
                # Optimal path to a leaf has been found
                return None
            
            num_potential_moves = len(current.get_potential_moves())
            num_children = 0 if current.children is None else len(current.children)
            if (num_potential_moves == 0):
                raise RuntimeError("During selection, found a node with no potential moves that is not a leaf.")
            
            if num_potential_moves > num_children:
                return self.expand(current)
            else: 
                print(f"finding max among current children for node: {current.get_loc()}")
                for child in current.children: 
                    print(f"Node has child: {child} with score {child.ucb_score()}")
                #return None 
                current = max(current.children, key=lambda child: child.ucb_score())
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        print(f"Expand called on {node}")
        explored = set()
        if node.children is not None:
            explored = {child.get_loc() for child in node.children}

        for path, loc in node.get_potential_moves():
            if not any(same_position(loc, e) for e in explored):
                # make a new child node at this location
                new_child = MonteCarloTree.Node(
                    tree=self,
                    depth=node.depth + 1,
                    loc=loc,
                    is_kernel=False,  # or True if you're treating kernels specially
                    path=path,
                    parent=node
                )
                if node.children is None:
                    node.children = []
                node.children.append(new_child)
                print(f"Added child: {new_child}")
                #file_name = f"map_from_{node.get_loc()[0]:.2f}_{node.get_loc()[1]:.2f}_to_{loc[0]:.2f}_{loc[1]:.2f}.png"
                #plot_move(self.shapely_map, self.shapely_obstacles, self.shapely_guard_positions[new_child.depth], Point(node.get_loc()), self.shadows[new_child.depth], Point(loc), path, new_child.depth-1, save_plot=True, file_name=file_name)
                return new_child
        
        raise RuntimeError("Expand was called on a fully-expanded node")

    def evaluate(self, node : MonteCarloTree.Node):
        cur_shadow = self._map._shadows[node.depth]
        guard_pos = self._guard.get_path()[node.depth]
        p_pt = Point(node.get_loc())
        
        if not cur_shadow.contains(p_pt):
            return 0.0 # score is 0 if the player is visible
        
        # find shortest distance to visible area
        vis_area = unary_union(self._map.get_visibility_polygons()[node.depth])
        shadow_score = self.alpha * p_pt.distance(vis_area)
        
        # find distance from the player to guard (length of shortest path)
        _, path_length = self._map.get_shortest_path(node.get_loc(), guard_pos)
        guard_distance_score = self.beta * path_length

        # find distance from the player to nearest kernel
        if (self._map._kernels):
            cur_kernels = self._map._kernels[node.depth]
            min_dist = math.inf
            for kernel in cur_kernels:
                try:
                    _, path_length = self._map.get_shortest_path(node.get_loc(), kernel.get_coords())
                    min_dist = min(min_dist, path_length)
                except Exception:
                    continue
            closest_kernel_score = self.gamma * min_dist
            
        node.score = shadow_score + guard_distance_score + closest_kernel_score
        return node.score
    
    def backpropagate(self, node : MonteCarloTree.Node, result):
        current = node
        while current is not None:
            current.visits += 1
            current.score += result
            current = current.parent
    
    def run(self):
        while True:
            selected = self.select()
            if (selected is None):
                print("Found optimal path!")
                #plot_paths(self)
                return
            
            result = self.evaluate(selected)
            self.backpropagate(selected, result)

    @staticmethod
    def traverse_and_plot(node: MonteCarloTree.Node):
        node.plot_move(save_dir='plots')
        if node.children:
            for child in node.children:
                MonteCarloTree.traverse_and_plot(child)

    class Node:
        '''
        Args: 
            depth: depth in the search tree of this node.
            loc: (x, y) coordinates of the point on the map this node represents.
            is_kernel: whether or not this node is a shadow kernel.
            kernel_depth: recursive depth of the kernel computation, if applicable, None otherwise. 
            path: path from the parent node to this node. Only None for the root node.
            parent: parent node, only None for the root node.
            children: list of child nodes, starts empty.
        '''
        def __init__(
                self, 
                tree : MonteCarloTree,
                depth : int,
                loc : Tuple[float, float], 
                is_kernel : bool,
                kernel_depth : Optional[int] = None, 
                path : Optional[LineString] = None,
                parent : Optional[MonteCarloTree.Node] = None,
                children : Optional[List[MonteCarloTree.Node]] = None,
                potential_moves : Optional[List[Tuple[LineString, Tuple[float, float]]]] = None,
            ):
            self.tree = tree
            self.depth = depth
            self.loc = loc
            self.potential_moves = potential_moves
            self.is_kernel = is_kernel
            self.kernel_depth = kernel_depth
            self.path = path
            self.parent = parent
            self.children = children

            self.score = 0.0
            self.visits = 0

            # compute if this node is in shadow?

        def ucb_score(self):
            if self.visits == 0:
                return float("inf")  # force exploration
            if (self.parent is None): # at the root
                return self.score/self.visits
            exploitation = self.score / self.visits
            exploration = MonteCarloTree.c * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
            return exploitation + exploration
        
        def __str__(self) -> str:
            return f"Node (loc=({self.loc[0]:.2f}, {self.loc[1]:.2f}), depth={self.depth})"
        
        def get_loc(self) -> Tuple[float, float]:
            return self.loc
        
        def get_potential_moves(self, prune_tol: float = 0.1) -> List[Tuple[LineString, Tuple[float, float]]]:
            if (self.potential_moves is None):
                moves = []
                if (self.depth == self.tree.max_depth):
                    return moves
                
                # Compute moves towards kernels
                # Map.Kernel.get_moves_towards_kernels()

                # Compute brute force moves
                num_directions = 6
                for i in range(num_directions):
                    angle = 2 * math.pi * i / num_directions
                    x = self.loc[0] + self.tree.get_max_step() * math.cos(angle)
                    y = self.loc[1] + self.tree.get_max_step() * math.sin(angle)
                    target = (x, y)
                    
                    # Try to find a valid path to this point
                    try:
                        path, length = self.tree.get_shortest_path(self.loc, target)
                        if path and length is not None:
                            line = LineString(path)
                            if length <= self.tree.get_max_step():
                                candidate = (line, target)
                            else:
                                # Take a point along the path at max_step distance
                                pt = line.interpolate(self.tree.get_max_step())
                                # Build truncated path
                                truncated_coords = []
                                dist_so_far = 0.0
                                for j in range(len(path) - 1):
                                    seg = LineString([path[j], path[j + 1]])
                                    seg_len = seg.length
                                    if dist_so_far + seg_len >= self.tree.get_max_step():
                                        remaining = self.tree.get_max_step() - dist_so_far
                                        cut_pt = seg.interpolate(remaining)
                                        truncated_coords.append((cut_pt.x, cut_pt.y))
                                        break
                                    else:
                                        truncated_coords.append(path[j + 1])
                                        dist_so_far += seg_len
                                truncated_line = LineString([path[0]] + truncated_coords)
                                candidate = (truncated_line, (pt.x, pt.y))
                            
                            moves.append(candidate)
                    except:
                        # Skip if path cannot be found (e.g., target is outside map or in obstacle)
                        continue
                
                # Prune moves: remove any that are within prune_tol of each other
                pruned_moves = []
                for candidate in moves:
                    too_close = False
                    for _, existing_pt in pruned_moves:
                        if math.dist(existing_pt, candidate[1]) < prune_tol:
                            too_close = True
                            break
                    if not too_close:
                        pruned_moves.append(candidate)
                
                self.potential_moves = pruned_moves

            return self.potential_moves
        
        def plot_move(self, plot_size = (11, 7), save_dir = 'plots'):
            plt.style.use("classic")

            print(f"Plotting move for node at depth {self.depth}, position ({self.loc[0]:.2f}, {self.loc[1]:.2f})")
            os.makedirs(save_dir, exist_ok=True)
            map = self.tree._map
        
            plt.figure(figsize=plot_size)
            ax = plt.gca()
            ax.axis("off")

            # Bounding box
            minx, miny, maxx, maxy = map._shapely_boundary.bounds
            bounds = box(minx - 1, miny - 1, maxx + 1, maxy + 1)
            ax.set_xlim(minx-1, maxx+1)
            ax.set_ylim(miny-1, maxy+1)

            # Obstacles union
            obstacles_union = unary_union(map._shapely_obstacles)

            # Outside region = bounding box minus map polygon
            outside = bounds.difference(map._shapely_boundary)

            # Draw outside region (non-walkable)
            add_polygon(ax, outside, fc="dimgray", ec="black", alpha=1.0, zorder=0)

            # Draw map interior (walkable area)
            add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0, zorder=1)

            # Draw obstacles inside map
            add_polygon(ax, obstacles_union, fc="dimgray", ec="black", alpha=1.0, zorder=2)

            # Shadow polygons
            add_polygon(ax, map._shadows[self.depth], fc="blue", alpha=0.3, zorder=3)

            # Guard position
            guard_pos = self.tree._guard.get_path()[self.depth]
            ax.plot(guard_pos[0], guard_pos[1], "r^", markersize=6, label="Guard")

            # Player position
            ax.plot(self.loc[0], self.loc[1], "bo", markersize=6, label="Player")


            #fig, ax = plt.subplots(figsize=(8, 8))
            #ax.set_aspect('equal', adjustable='box')

            if self.parent is not None:
                parent_pt = self.parent.get_loc()
                ax.plot(*parent_pt, 'ro', markersize=6, label='Parent')
                if self.path is not None:
                    x, y = self.path.xy
                    ax.plot(x, y, 'g--', linewidth=1.5, label='Path from parent')

            ax.set_title(
                f"Depth: {self.depth} | Score: {self.score/self.visits:.3f} | Visits: {self.visits}",
                fontsize=11
            )
            ax.legend(loc='upper right', fontsize=8)

            # --- 6️⃣ Save to file ---
            filename = f"move_depth_{self.depth:03d}_x{self.loc[0]:.2f}_y{self.loc[1]:.2f}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=200)
            plt.close()
            print(f"✅ Saved move plot to {filepath}")