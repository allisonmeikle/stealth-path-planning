from __future__ import annotations

import math
import time
import random

from typing import List, Optional, Tuple
from shapely.geometry import Point, LineString

from characters import *
from map import Map
from plot_helper import *
from helpers import *

class MonteCarloTree:
    ''' Hyperparameters - Tunable! '''
    _c = math.sqrt(2.0) # for ucb score (constant)

    _k = 1.0 # for progressive widening (constant)
    _alpha = 0.05 # for progressive widening (exponent)

    _delta = 1.0 # for shadow score
    _beta = 0.4 # for guard distance score
    _gamma = -0.5 # for kernel distance score
    _eta = 1.0   # depth bonus weight

    _lambda = 3.0 # weight of rollout value in heuristic
    _phi = 20 # rollout depth

    _brute_force = False # using brute force moves

    def __init__(self, map: Map, max_edges = 5, debug = False):
        self._map = map
        self._max_edges = max_edges
        self._debug = debug
        self._transposition_table = {}
        self._root = MonteCarloTree.Node(self, map.get_player_start_pos(), 0)
        self._max_depth = map.get_num_timesteps()-1
        self._saw_terminal = False
        
        # Tree stats
        self._num_nodes = 0
        self._depth = 0
        self._breadth = {0: 1}

    def select(self) -> Tuple[MonteCarloTree.Node, List[MonteCarloTree.Edge]]:
        node = self._root
        path_edges = []

        while True:
            if (node._depth >= self._max_depth):
                return node, path_edges
            
            moves = node.get_potential_moves()
            if (len(moves) == 0):
                return node, path_edges
            
            allowed_children = min(len(moves), max(1, MonteCarloTree._k * (node._visits ** MonteCarloTree._alpha)))
            if self._debug:
                print(f"Allowed {allowed_children} children at depth {self._depth}")
            if len(node._edges) < min(self._max_edges, allowed_children):
                return self.expand(node), path_edges

            # choose best edge
            best_edge = max(node._edges, key=lambda e: e.ucb_score())
            path_edges.append(best_edge)
            node = best_edge._child
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        used_locs = {e._child.get_loc() for e in node._edges}
        moves = node.get_potential_moves()
        for (path, loc) in moves:
            if loc not in used_locs:
                child = self._get_or_create_node(loc, node._depth + 1)
                edge = MonteCarloTree.Edge(node, child, path)
                node._edges.append(edge)
                return child

        raise RuntimeError("Expand called on fully-expanded node")
        
    def rollout(self, node: MonteCarloTree.Node):
        cur_loc = node._loc
        cur_depth = node._depth

        for _ in range(MonteCarloTree._phi):
            if cur_depth >= self._max_depth:
                return 1.0

            fake_node = self._get_or_create_node(cur_loc, cur_depth, store=False)
            moves = fake_node.get_potential_moves()
            shadow = self._map.get_shadow(cur_depth)
            if not moves or not shadow.contains(Point(cur_loc)):
                return 0.0

            # greedily pick move towards best kernel at this time
            _, next_loc = moves[0]
            cur_loc = next_loc
            cur_depth += 1
        
        return 1.0

    def evaluate(self, node : MonteCarloTree.Node):
        return node.get_score() + MonteCarloTree._lambda*self.rollout(node)
    
    def backpropagate(self, leaf : MonteCarloTree.Node, path_edges : List[MonteCarloTree.Edge], result):
        leaf._visits += 1
        leaf._total_value += result
        if leaf._depth == self._max_depth:
            self._saw_terminal = True

        for edge in reversed(path_edges):
            edge._visits += 1
            edge._total_value += result

            parent = edge._parent
            parent._visits += 1
            parent._total_value += result
    
    def run(self, total_time):
        start = time.time()

        while True:
            now = time.time()
            elapsed = now - start

            # stop if full time budget is completed
            if elapsed >= total_time:
                return

            # normal MCTS iteration
            leaf, edges = self.select()
            value = self.evaluate(leaf)
            self.backpropagate(leaf, edges, value)

    def get_map(self):
        return self._map
    
    def _get_or_create_node(self, loc, depth, store=True):
        qx, qy = self._map.quantize_point(loc)
        key = (qx, qy, depth)
        if key in self._transposition_table:
            return self._transposition_table[key]
        node = MonteCarloTree.Node(self, loc, depth)
        self._transposition_table[key] = node
        # only update stats for expanded nodes, not rollouts
        if store:
            self._num_nodes += 1
            self._depth = max(self._depth, depth)
            if depth not in self._breadth:
                self._breadth[depth] = 0
            self._breadth[depth] += 1
        return node
    
    def get_best_path(self):
        node = self._root
        path = []

        while node._edges:
            best_edge = max(node._edges, key=lambda e: e._visits)
            path.append(best_edge)
            node = best_edge._child

        return path

    def get_stats(self):
        stats = f"Hyperparameters:\n k={MonteCarloTree._k} (progressive widening constant)\n"
        stats += f"c={MonteCarloTree._c} (exploration constant)\n"
        stats += f"alpha={MonteCarloTree._alpha} (progressive widening exponent)\n"
        stats += f"beta={MonteCarloTree._beta} (guard distance score)\n"
        stats += f"delta={MonteCarloTree._delta} (distance to visible area score)\n"
        stats += f"gamma={MonteCarloTree._gamma} (kernel distance score)\n"
        stats += f"eta={MonteCarloTree._eta} (depth bonus)\n"
        stats += f"lambda={MonteCarloTree._lambda} (weight of rollout value in heuristic)\n"
        stats += f"phi={MonteCarloTree._phi} (rollout depth)\n"  
        stats += f"using brute force moves = {MonteCarloTree._brute_force}\n"             

        stats += f"Total nodes in tree: {self._num_nodes}, with max depth {self._depth}, and maximum allowed edges {self._max_edges}\n"
        stats += "Node breadth by depth:\n"
        for depth in sorted(self._breadth):
            stats += f"  Depth {depth}: {self._breadth[depth]} node(s)\n"
                                                                  
        return stats

    class Node:
        def __init__(self, tree : MonteCarloTree, loc : Tuple[float, float], depth : int):
            self._tree = tree
            self._loc = loc
            self._depth = depth
            self._map = tree.get_map()

            self._potential_moves: Optional[List[Tuple[LineString, Tuple[float, float]]]] = None
            self._edges: List[MonteCarloTree.Edge] = []
            self._score: Optional[float] = None
            self._total_value = 0.0
            self._visits = 0.0
        
        def get_loc(self) -> Tuple[float, float]:
            return self._loc
        
        @staticmethod
        def sigmoid(x: float):
            return 1 / (1 + math.exp(-x))

        def get_score(self):
            if (not self._score):
                shadow = self._map.get_shadow(self._depth)
                guard_positions = self._map.get_guard_positions(self._depth)
                pt = Point(self._loc)
                
                if not shadow.contains(pt):
                    self._score = 0.0 # score is 0 if the player is visible
                    return self._score
            
                # find shortest distance to visible area
                vis_area = self._map.get_visibility_polygon(self._depth)
                shadow_score = MonteCarloTree._delta * MonteCarloTree.Node.sigmoid(pt.distance(vis_area))
                
                # find distance from the player to guard (length of shortest path)
                shortest_path = math.inf
                for pos in guard_positions:
                    _, path_length = self._map.get_shortest_path(self._loc, pos)
                    shortest_path = min(shortest_path, path_length)
                guard_distance_score = MonteCarloTree._beta * MonteCarloTree.Node.sigmoid(shortest_path)

                # find distance from the player to nearest kernel
                if (self._map._kernels):
                    cur_kernels = self._map._kernels[self._depth]
                    min_dist = math.inf
                    for kernel in cur_kernels:
                        try:
                            _, path_length = self._map.get_shortest_path(self._loc, kernel.get_coords())
                            min_dist = min(min_dist, path_length)
                        except Exception:
                            continue
                    closest_kernel_score = MonteCarloTree._gamma * MonteCarloTree.Node.sigmoid(min_dist)

                # depth bonus 
                depth_ratio = self._depth / self._tree._max_depth
                depth_bonus = MonteCarloTree._eta * depth_ratio
                    
                self._score = shadow_score + guard_distance_score + closest_kernel_score + depth_bonus
            return self._score
                
        def get_potential_moves(self, prune_tol: float = 0.1) -> List[Tuple[LineString, Tuple[float, float]]]:
            if (self._potential_moves is None):
                moves = []
                if (self._depth == self._tree._max_depth):
                    return moves
                
                # Compute moves towards kernels
                moves.extend(self.get_moves_towards_kernels())

                # Compute brute force moves
                if (MonteCarloTree._brute_force):
                    moves.extend(self.get_brute_force_moves())
                
                # Prune moves: remove any that are within prune_tol of each other
                pruned_moves = []
                for candidate in moves:
                    too_close = False
                    for _, existing_pt in pruned_moves:
                        if self._map.is_same_position(existing_pt, candidate[1]):
                            too_close = True
                            break
                    if not too_close:
                        pruned_moves.append(candidate)
                
                self._potential_moves = pruned_moves
                if (self._tree._debug):
                    print(f"Got {len(self._potential_moves)} at depth {self._depth}")
            return self._potential_moves
        
        def get_brute_force_moves(self, num_directions = 4) -> List[Tuple[LineString, Tuple[float, float]]]:
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
                        if self._map.is_valid_position(candidate[1]):
                            moves.append(candidate)
                except:
                    # Skip if path cannot be found (e.g., target is outside map or in obstacle)
                    continue
            random.shuffle(moves)
            return moves
            
        def get_moves_towards_kernels(self):
            moves = []

            for kernel in self._map.get_kernels(self._depth+1):
                if same_position(self._loc, kernel.get_coords()):
                    continue
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
                    if self._map.is_valid_position((pt.x, pt.y)):
                        moves.append((truncated_line, (pt.x, pt.y)))
            return moves

        def __str__(self) -> str:
            return f"Node (loc=({self._loc[0]:.2f}, {self._loc[1]:.2f}), depth={self._depth}, score {self._total_value/self._visits:.2f})"

    class Edge: 
        def __init__(self, parent: MonteCarloTree.Node, child: MonteCarloTree.Node, path: LineString):
            self._parent = parent
            self._child = child
            self._path = path

            self._visits = 0
            self._total_value = 0.0
            px, py = parent._loc
            cx, cy = child._loc
            dx, dy = cx - px, cy - py
            length = math.hypot(dx, dy)
            self._direction = (0.0, 0.0)
            if length > 0:
                self._direction = (dx / length, dy / length)
                
        def ucb_score(self):
            if self._visits == 0:
                return float("inf")
            else:
                exploitation = self._total_value / self._visits
                exploration = MonteCarloTree._c * math.sqrt(
                    math.log(self._parent._visits + 1) / self._visits
                )
                return exploitation + exploration

        def __str__(self) -> str:
            return f"Edge (score={self._total_value/self._visits:.2f}"