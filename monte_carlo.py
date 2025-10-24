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

    def __init__(self, map: Map):
        self._map = map
        self.root = MonteCarloTree.Node(self, 0, map.get_player_start_pos(), map)
        self.max_depth = map.get_num_timesteps()-1

        # Score hyperparameters
        self.alpha = 1 # for shadow score
        self.beta = 1 # for guard distance score
        # maybe scale this down, wont change a lot with small steps
        self.gamma = -1 # for kernel distance score
        
    def select(self) -> Optional[MonteCarloTree.Node]:
        current = self.root
        while True:
            print("Current node in selection loop: ", current)

            if (current._depth >= self.max_depth):
                print("Selection ended on a leaf, returning None")
                # Optimal path to a leaf has been found
                return None
            
            num_potential_moves = len(current.get_potential_moves())
            num_children = 0 if current._children is None else len(current._children)
            if (num_potential_moves == 0):
                raise RuntimeError("During selection, found a node with no potential moves that is not a leaf.")
            
            if num_potential_moves > num_children:
                return self.expand(current)
            else: 
                print(f"finding max among current children for node: {current.get_loc()}")
                for child in current._children: 
                    print(f"Node has child: {child} with score {child.ucb_score()}")
                #return None 
                current = max(current._children, key=lambda child: child.ucb_score())
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        print(f"Expand called on {node}")
        explored = set()
        if node._children is not None:
            explored = {child.get_loc() for child in node._children}

        for path, loc in node.get_potential_moves():
            if not any(same_position(loc, e) for e in explored):
                # make a new child node at this location
                new_child = MonteCarloTree.Node(
                    tree=self,
                    depth=node._depth + 1,
                    loc=loc,
                    map = self._map,
                    path=path,
                    parent=node
                )
                if node._children is None:
                    node._children = []
                node._children.append(new_child)
                print(f"Added child: {new_child}")
                #file_name = f"map_from_{node.get_loc()[0]:.2f}_{node.get_loc()[1]:.2f}_to_{loc[0]:.2f}_{loc[1]:.2f}.png"
                #plot_move(self.shapely_map, self.shapely_obstacles, self.shapely_guard_positions[new_child.depth], Point(node.get_loc()), self.shadows[new_child.depth], Point(loc), path, new_child.depth-1, save_plot=True, file_name=file_name)
                return new_child
        
        raise RuntimeError("Expand was called on a fully-expanded node")

    def evaluate(self, node : MonteCarloTree.Node):
        cur_shadow = self._map._shadows[node._depth]
        guard_positions = self._map.get_guard_positions(node._depth)
        p_pt = Point(node.get_loc())
        
        if not cur_shadow.contains(p_pt):
            return 0.0 # score is 0 if the player is visible
        
        # find shortest distance to visible area
        vis_area = unary_union(self._map.get_visibility_polygons()[node._depth])
        shadow_score = self.alpha * p_pt.distance(vis_area)
        
        # find distance from the player to guard (length of shortest path)
        shortest_path = math.inf
        for pos in guard_positions:
            _, path_length = self._map.get_shortest_path(node.get_loc(), pos)
            shortest_path = min(shortest_path, path_length)
        guard_distance_score = self.beta * shortest_path

        # find distance from the player to nearest kernel
        if (self._map._kernels):
            cur_kernels = self._map._kernels[node._depth]
            min_dist = math.inf
            for kernel in cur_kernels:
                try:
                    _, path_length = self._map.get_shortest_path(node.get_loc(), kernel.get_coords())
                    min_dist = min(min_dist, path_length)
                except Exception:
                    continue
            closest_kernel_score = self.gamma * min_dist
            
        node._score = shadow_score + guard_distance_score + closest_kernel_score
        return node._score
    
    def backpropagate(self, node : MonteCarloTree.Node, result):
        current = node
        while current is not None:
            current._num_visits += 1
            current._score += result
            current = current._parent
    
    def run(self):
        while True:
            selected = self.select()
            if (selected is None):
                print("Found optimal path!")
                best_leaf = self.get_best_leaf()
                if best_leaf and best_leaf._num_visits == 0:
                    result = self.evaluate(best_leaf)
                    self.backpropagate(best_leaf, result)
                return
                #plot_paths(self)
                return
            
            result = self.evaluate(selected)
            self.backpropagate(selected, result)

    @staticmethod
    def traverse_and_plot(node: MonteCarloTree.Node):
        node.plot_move(save_dir='plots')
        if node._children:
            for child in node._children:
                MonteCarloTree.traverse_and_plot(child)

    def get_best_leaf(self) -> Optional["MonteCarloTree.Node"]:
        """
        Traverse the entire tree to find the leaf node with the highest average score.
        """
        best_node = None
        best_score = -math.inf

        def dfs(node: "MonteCarloTree.Node"):
            nonlocal best_node, best_score
            if not node._children:  # leaf
                if node._num_visits > 0:
                    avg_score = node._score / node._num_visits
                    if avg_score > best_score:
                        best_score = avg_score
                        best_node = node
            else:
                for child in node._children:
                    dfs(child)

        dfs(self.root)
        print(f"🌟 Best leaf at depth {best_node._depth} with score {best_score:.3f}")
        return best_node

    class Node:
        def __init__(
                self, 
                tree : MonteCarloTree,
                depth : int,
                loc : Tuple[float, float], 
                map : Map,
                path : Optional[LineString] = None,
                parent : Optional[MonteCarloTree.Node] = None,
                children : Optional[List[MonteCarloTree.Node]] = None,
                potential_moves : Optional[List[Tuple[LineString, Tuple[float, float]]]] = None,
            ):
            self._tree = tree
            self._depth = depth
            self._loc = loc
            self._map = map
            self._potential_moves = potential_moves
            self._parent = parent
            self._path_from_parent = path
            self._children = children

            self._score = 0.0
            self._num_visits = 0

        def ucb_score(self):
            if self._num_visits == 0:
                return float("inf")  # force exploration
            if (self._parent is None): # at the root
                return self._score/self._num_visits
            exploitation = self._score / self._num_visits
            exploration = MonteCarloTree.c * math.sqrt(
                math.log(self._parent._num_visits) / self._num_visits
            )
            return exploitation + exploration
        
        def __str__(self) -> str:
            return f"Node (loc=({self._loc[0]:.2f}, {self._loc[1]:.2f}), depth={self._depth})"
        
        def get_loc(self) -> Tuple[float, float]:
            return self._loc
        
        def get_potential_moves(self, prune_tol: float = 0.1) -> List[Tuple[LineString, Tuple[float, float]]]:
            if (self._potential_moves is None):
                moves = []
                print(f"Computing potential moves for node {self}")
                if (self._depth == self._tree.max_depth):
                    return moves
                
                # Compute moves towards kernels
                moves.extend(self.get_moves_towards_kernels())

                # Compute brute force moves
                moves.extend(self.get_brute_force_moves())
                
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
                
                self._potential_moves = pruned_moves
            return self._potential_moves
        
        def get_brute_force_moves(self, num_directions = 6) -> List[Tuple[LineString, Tuple[float, float]]]:
            moves = []
            for i in range(num_directions):
                angle = 2 * math.pi * i / num_directions
                x = self._loc[0] + self._map.get_player_max_step() * math.cos(angle)
                y = self._loc[1] + self._map.get_player_max_step() * math.sin(angle)
                target = (x, y)
                
                # Try to find a valid path to this point
                try:
                    path, length = self._map.get_shortest_path(self._loc, target)
                    if path and length is not None:
                        line = LineString(path)
                        if length <= self._map.get_player_max_step():
                            candidate = (line, target)
                        else:
                            # Take a point along the path at max_step distance
                            pt = line.interpolate(self._map.get_player_max_step())
                            # Build truncated path
                            truncated_coords = []
                            dist_so_far = 0.0
                            for j in range(len(path) - 1):
                                seg = LineString([path[j], path[j + 1]])
                                seg_len = seg.length
                                if dist_so_far + seg_len >= self._map.get_player_max_step():
                                    remaining = self._map.get_player_max_step() - dist_so_far
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
                return moves
            
        def get_moves_towards_kernels(self):
            moves = []
            for kernel in self._map.get_kernels(self._depth+1):
                target = kernel.get_coords()
                path, length = self._map.get_shortest_path(self._loc, target)
                if not path or not length:
                    continue

                line = LineString(path)
                if length <= self._map.get_player_max_step():
                    # Entire kernel is reachable
                    moves.append((line, target))
                else:
                    # Take a point along the path at max_step distance
                    pt = line.interpolate(self._map.get_player_max_step())
                    # Build truncated path: from start → pt
                    truncated_coords = []
                    dist_so_far = 0.0
                    for i in range(len(path) - 1):
                        seg = LineString([path[i], path[i + 1]])
                        seg_len = seg.length
                        if dist_so_far + seg_len >= self._map.get_player_max_step():
                            # Cut inside this segment
                            remaining = self._map.get_player_max_step() - dist_so_far
                            cut_pt = seg.interpolate(remaining)
                            truncated_coords.append((cut_pt.x, cut_pt.y))
                            break
                        else:
                            truncated_coords.append(path[i + 1])
                            dist_so_far += seg_len
                    truncated_line = LineString([path[0]] + truncated_coords)
                    moves.append((truncated_line, (pt.x, pt.y)))
            return moves
        
        def plot_move(self, plot_size = (11, 7), save_dir = 'plots_new'):
            plt.style.use("classic")

            print(f"Plotting move for node at depth {self._depth}, position ({self._loc[0]:.2f}, {self._loc[1]:.2f})")
            os.makedirs(save_dir, exist_ok=True)
            map = self._tree._map
        
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
            add_polygon(ax, map._shadows[self._depth], fc="blue", alpha=0.3, zorder=3)

            # Guard position
            guard_pos = self._tree._guard.get_path()[self._depth]
            ax.plot(guard_pos[0], guard_pos[1], "r^", markersize=6, label="Guard")

            # Player position
            ax.plot(self._loc[0], self._loc[1], "bo", markersize=6, label="Player")


            #fig, ax = plt.subplots(figsize=(8, 8))
            #ax.set_aspect('equal', adjustable='box')

            if self._parent is not None:
                parent_pt = self._parent.get_loc()
                ax.plot(*parent_pt, 'ro', markersize=6, label='Parent')
                if self._path_from_parent is not None:
                    x, y = self._path_from_parent.xy
                    ax.plot(x, y, 'g--', linewidth=1.5, label='Path from parent')

            ax.set_title(
                f"Depth: {self._depth} | Score: {self._score/self._num_visits:.3f} | Visits: {self._num_visits}",
                fontsize=11
            )
            ax.legend(loc='upper right', fontsize=8)

            # --- 6️⃣ Save to file ---
            filename = f"move_depth_{self._depth:03d}_x{self._loc[0]:.2f}_y{self._loc[1]:.2f}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=200)
            plt.close()
            print(f"✅ Saved move plot to {filepath}")