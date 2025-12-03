from __future__ import annotations

import math
import time
import random

from typing import List, Optional, Tuple
from shapely.geometry import Point, LineString

from characters import *
from map import Map
from helpers import *

class MonteCarloTree:
    ''' Hyperparameters - Tunable! '''
    _c = math.sqrt(2.0) # for ucb score (constant)

    _k = 1.0 # for progressive widening (constant)
    _alpha = 0.2 # for progressive widening (exponent)

    _delta = 1.0 # for shadow score
    _beta = 0.4 # for guard distance score
    _gamma = 1.0 # for kernel distance score
    _tau = 0.1 # for distance-decay rate
    _top_k = 3 # how many kernels to path plan to
    _eta = 0.0   # depth bonus weight

    _lambda = 3.0 # weight of rollout value in heuristic
    _phi = 20 # rollout depth
    _rollout_ratio = False # True if using survived steps/rollout_depths for score, False is using 0 or 1

    _brute_force = False # using brute force moves

    def __init__(self, map: Map, max_edges = 10):
        self._map = map
        self._max_edges = max_edges
        self._nodes = []
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
            
            allowed_children = min(len(moves), max(1, math.floor(MonteCarloTree._k * (node._visits ** MonteCarloTree._alpha))))
            if len(node._outgoing_edges) < min(self._max_edges, allowed_children):
                new_child = self.expand(node)
                if new_child is not None:
                    return new_child, path_edges

            # choose best edge
            best_edge = max(node._outgoing_edges, key=lambda e: e.ucb_score())
            path_edges.append(best_edge)
            node = best_edge._child
    
    def expand(self, node : MonteCarloTree.Node) -> Optional[MonteCarloTree.Node]:
        used_locs = {self._map.quantize_point(e._child.get_loc()) for e in node._outgoing_edges}
        for target in node.get_potential_moves():
            #print(f"Got potential move {act_type} to {target}")
            qtarget = self._map.quantize_point(target)
            if qtarget in used_locs:
                continue
            res = self._map.get_longest_move_along_shortest_path(node._loc, target)
            if res is None:
                continue
            path, new_loc = res
            if len(path) == 1:
                path, new_loc = [node._loc, target], target
                
            qnew = node._map.quantize_point(new_loc)
            if qnew in used_locs:
                continue
            #print(f"Got path from {node._loc} to {new_loc} along path {path}")
            # make child
            child = self._get_or_create_node(new_loc, node._depth + 1)
            edge = MonteCarloTree.Edge(node, child, LineString(path))
            node._outgoing_edges.append(edge)
            child._incoming_edges.append(edge)
            return child

        return None  # no new moves
        
    def rollout(self, node: MonteCarloTree.Node):
        cur_loc = node._loc
        cur_depth = node._depth
        max_step = self._map.get_player_max_step()
        survived_steps = 0
        last_hidden = False

        for _ in range(MonteCarloTree._phi):
            # check last state visibility
            last_hidden = not self._map.is_visible(cur_loc, cur_depth)
            if last_hidden:
                survived_steps += 1

            # early stop if we reached the final guard timestep
            if cur_depth >= self._max_depth:
                break

            # random rollout step
            '''            while True:
                angle = random.random() * 2 * math.pi
                dist = random.random() * max_step
                candidate = (
                    cur_loc[0] + dist * math.cos(angle),
                    cur_loc[1] + dist * math.sin(angle)
                )
                if self._map.is_valid_position(candidate):
                    cur_loc = candidate
                    break
            '''

            cur_depth += 1

        if MonteCarloTree._rollout_ratio:
            return survived_steps / MonteCarloTree._phi 
        # otherwise return 1 if last state hidden, else 0
        return 1.0 if last_hidden else 0.0

    def evaluate(self, node : MonteCarloTree.Node):
        return node.get_score()
    
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
            #if elapsed >= total_time and self._depth >= self._max_depth:
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
            self._nodes.append(node)
        return node
    
    def get_path_max_score(self):
        node = self._root
        path = []

        while node._outgoing_edges:
            best_edge = max(node._outgoing_edges, key=lambda e: e._total_value / e._visits if e._visits > 0 else float("-inf"))
            path.append(best_edge)
            node = best_edge._child

        return path
    
    def get_path_max_visits(self):
        node = self._root
        path = []
        while node._outgoing_edges:
            best = max(node._outgoing_edges, key=lambda e: e._visits)
            path.append(best)
            node = best._child
        return path
    
    def get_path_backtracked(self):
        # find best full-depth leaf
        leaves = [n for n in self._nodes if n._depth == self._depth]
        if not leaves:
            return []  # fallback
        
        best_leaf = max(
            leaves,
            key=lambda n: (n._total_value / n._visits) if n._visits>0 else 0
        )

        # backtrack
        path = []
        node = best_leaf
        while node._incoming_edges:
            best_edge = max(
                node._incoming_edges,
                key=lambda e: e._total_value / e._visits if e._visits > 0 else -1e9
            )

            path.append(best_edge)
            node = best_edge._parent  # move up to parent node
        path.reverse()
        return path

    def get_stats(self):
        stats = f"Hyperparameters:\n k={MonteCarloTree._k} (progressive widening constant)\n"
        stats += f"c={MonteCarloTree._c} (exploration constant)\n"
        stats += f"alpha={MonteCarloTree._alpha} (progressive widening exponent)\n"
        stats += f"beta={MonteCarloTree._beta} (guard distance score)\n"
        stats += f"delta={MonteCarloTree._delta} (distance to visible area score)\n"
        stats += f"gamma={MonteCarloTree._gamma} (kernel distance score)\n"
        stats += f"tau={MonteCarloTree._tau} (kernel distance-decay rate)\n"
        stats += f"top_k={MonteCarloTree._top_k} (num of kernels evaluated for score)\n"
        stats += f"eta={MonteCarloTree._eta} (depth bonus)\n"
        stats += f"lambda={MonteCarloTree._lambda} (weight of rollout value in heuristic)\n"
        stats += f"phi={MonteCarloTree._phi} (rollout depth)\n"  
        stats += f"using rollout ratio={MonteCarloTree._rollout_ratio}\n"  
        stats += f"using brute force moves = {MonteCarloTree._brute_force}\n"             

        stats += f"Total nodes in tree: {self._num_nodes}, with max depth {self._depth}, and maximum allowed edges {self._max_edges}\n"
        stats += "Node breadth by depth:\n"
        for depth in sorted(self._breadth):
            stats += f"  Depth {depth}: {self._breadth[depth]} node(s)\n"

        stats += f"Path cache hit ratio {self._map._path_cache_hits / self._map._path_cache_queries}, {self._map._path_cache_hits} hits, {self._map._path_cache_queries} queries\n"
                                                                  
        return stats

    class Node:
        def __init__(self, tree : MonteCarloTree, loc : Tuple[float, float], depth : int):
            self._tree = tree
            self._loc = loc
            self._depth = depth
            self._map = tree.get_map()

            self._potential_moves: Optional[List[Tuple[LineString, Tuple[float, float]]]] = None
            self._incoming_edges: List[MonteCarloTree.Edge] = []
            self._outgoing_edges: List[MonteCarloTree.Edge] = []
            self._score: Optional[float] = None
            self._total_value = 0.0
            self._visits = 0.0
        
        def get_loc(self) -> Tuple[float, float]:
            return self._loc
        
        @staticmethod
        def sigmoid(x: float):
            return 1 / (1 + math.exp(-x))

        def get_score(self):
            if (self._score is None):
                if self._map.is_visible(self._loc, self._depth):
                    self._score = 0.0 # score is 0 if the player is visible
                    return self._score
                
                # find shortest distance to visible area
                vis_area = self._map.get_visibility_polygon(self._depth)
                d = Point(self._loc).distance(vis_area)
                #shadow_score = MonteCarloTree._delta * MonteCarloTree.Node.sigmoid(d)
                #shadow_val = (d/(1.0+d))
                shadow_val = 1.0 - math.exp(-MonteCarloTree._tau * d)
                #print(f"Shadow distance {d}, gave shadow val {shadow_val}")
                shadow_score = MonteCarloTree._delta * shadow_val
            
                #print(f"Got shadow score {shadow_score}")

                # find distance from the player to guard (length of shortest path)
                shortest_path = math.inf
                guard_positions = self._map.get_guard_positions(self._depth)
                for pos in guard_positions:
                    _, path_length = self._map.get_shortest_path(self._loc, pos)
                    shortest_path = min(shortest_path, path_length)
                #guard_score = (shortest_path/(1.0+shortest_path))
                guard_score = 1.0 - math.exp(-MonteCarloTree._tau * shortest_path)
                #print(f"Guard distance {shortest_path} gave guard score {guard_score}")
                #guard_distance_score = MonteCarloTree._beta * MonteCarloTree.Node.sigmoid(shortest_path)
                guard_distance_score = MonteCarloTree._beta * (guard_score)
                

                # find distance from the player to the 3 deepest kernels
                kernels = self._map.get_kernels(self._depth)
                top_k = kernels[:MonteCarloTree._top_k] if len(kernels) >= MonteCarloTree._top_k else kernels[:]
                path_dists = []
                weights = []
                for ker in top_k:
                    _, path_length = self._map.get_shortest_path(self._loc, ker.get_coords())
                    path_dists.append(path_length)
                    weights.append(ker.get_depth() + 1)
                #avg_dist = sum(w * d for (w,d) in zip(weights, path_dists)) / sum(weights)
                avg_dist = sum(path_dists)/(len(path_dists))
                #score = 1.0/(1.0+avg_dist)
                score = math.exp(-MonteCarloTree._tau * avg_dist)
                #score = (avg_dist/(1.0+avg_dist))
                #print(f"Average path distance {avg_dist} gave score {score}")
                #print(f"Normalized path distance: {norm}, for distances {dist for dist in path_dists}")
                kernel_score = MonteCarloTree._gamma * score

                #print(f"path_dist {path_dist}")
                #kernel_score = MonteCarloTree._gamma * (1 / (1 + path_dist / 5.0))
                #kernel_score = MonteCarloTree._gamma * math.exp(-path_dist / MonteCarloTree._tau)
                #print(f"Kernel at depth {self._depth}, loc {self._loc} got kernel score {kernel_score}")

                # depth bonus 
                depth_bonus = MonteCarloTree._eta * (self._depth / self._tree._max_depth)

                # rollout score
                rollout_score = MonteCarloTree._lambda * self._tree.rollout(self)
                    
                self._score = shadow_score + guard_distance_score + kernel_score + depth_bonus + rollout_score
            return self._score
                
        def get_potential_moves(self) -> List[Tuple[LineString, Tuple[float, float]]]:
            if self._potential_moves is None:
                moves = []
                if (self._depth == self._tree._max_depth or self._map.is_visible(self._loc, self._depth)):
                    return moves
                
                # Compute moves towards kernels
                moves.extend(self.get_moves_towards_kernels())
                # Compute brute force moves
                if (MonteCarloTree._brute_force):
                    moves.extend(self.get_brute_force_moves())
                self._potential_moves = moves

            return self._potential_moves
        
        def get_brute_force_moves(self, num_directions=4):
            moves = []
            max_step = self._map.get_player_max_step()
            x0, y0 = self._loc
            for i in range(num_directions):
                angle = 2 * math.pi * i / num_directions
                target = (x0 + max_step * math.cos(angle), y0 + max_step * math.sin(angle))
                moves.append(target)
            random.shuffle(moves)
            return moves

        def get_moves_towards_kernels(self):
            return [kernel.get_coords() for kernel in self._map.get_kernels(self._depth + 1)]
            
            '''
            moves = []

            for kernel in self._map.get_kernels(self._depth+1):
                if self._map.is_same_position(self._loc, kernel.get_coords()):
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
            '''

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