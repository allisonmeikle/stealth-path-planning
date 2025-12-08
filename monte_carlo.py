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
    _alpha = 0.3 # for progressive widening (exponent)

    _delta = 0.5 # for visible area distance score
    _beta = 0.4 # for guard distance score
    _gamma = 1.0 # for kernel distance score
    _theta = 1.0 # for shadow distance score (for visible nodes)
    _tau = 0.1 # for distance-decay rate
    _top_k = 1 # how many kernels to path plan to in scoring
    _eta = 5.0   # depth bonus weight

    _lambda = 3.0 # weight of rollout value in heuristic
    _phi = 20 # rollout depth
    _rollout_ratio = False # True if using survived steps/rollout_depths for score, False is using 0 or 1

    _visible_node_penalty = -100 # Score given to positions visible by a guard
    _visible_edge_penalty = 0.1
    _max_kernel_moves = math.inf # maximum number of moves path planning to a kernel for each node 
    _brute_force = False # using brute force moves

    def __init__(self, map: Map, max_edges = 10):
        self._map = map
        self._max_edges = max_edges
        self._nodes = []
        self._transposition_table = {}
        self._tt_queries = 0
        self._tt_hits = 0
        self._root = MonteCarloTree.Node(self, map.get_player_start_pos(), 0)
        self._max_depth = map.get_num_timesteps()-1
        self._saw_terminal = False
        
        # Tree stats
        self._num_nodes = 0
        self._depth = 0
        self._breadth = {0: 1}
        
        self._score_cache = {}

    def select(self) -> Tuple[MonteCarloTree.Node, List[MonteCarloTree.Edge]]:
        node = self._root
        path_edges = []

        while True:            
            moves = node.get_potential_moves()            
            allowed_children = min(len(moves), max(1, math.floor(MonteCarloTree._k * (node._visits ** MonteCarloTree._alpha))))
            needs_expansion = not node._fully_expanded and len(node._outgoing_edges) < min(self._max_edges, allowed_children)
            if len(moves) == 0 or needs_expansion or self._map.is_visible(node._loc, node._depth):
                # Reached a "leaf" or a node that needs expansion
                return node, path_edges

            # choose best edge
            best_edge = max(node._outgoing_edges, key=lambda e: e.ucb_score())
            path_edges.append(best_edge)
            node = best_edge._child
    
    def expand(self, node : MonteCarloTree.Node) -> Optional[Tuple[MonteCarloTree.Node, MonteCarloTree.Edge]]:
        if node._fully_expanded:
            return None
        
        used_locs = {e._child.get_loc() for e in node._outgoing_edges}
        moves = node.get_potential_moves()
        #print(f"Expanding {node} that has {len(moves)} potential moves and {len(node._outgoing_edges)} outgoing edges")
        for target in moves:
            res = self._map.get_longest_move_along_shortest_path(node._loc, target)
            if res is None:
                continue
            path, new_loc = res
            if len(path) == 1:
                path, new_loc = [node._loc, target], target
                
            if any(self._map.is_same_position(new_loc, uloc) for uloc in used_locs):
                continue

            # make child
            #print(f"Adding node at {new_loc} at depth {node._depth + 1}")
            child = self._get_or_create_node(new_loc, node._depth + 1)
            edge = MonteCarloTree.Edge(node, child, LineString(path), self._map.get_visibility_polygon(node._depth))
            node._outgoing_edges.append(edge)
            child._incoming_edges.append(edge)
            return child, edge
        node._fully_expanded = True
        return None  # no new moves
        
    def rollout(self, node: MonteCarloTree.Node):
        cur_loc = node._loc
        cur_depth = node._depth
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
        for edge in reversed(path_edges):
            edge._visits += 1
            edge._total_value += result
            parent = edge._parent
            parent._visits += 1
            parent._total_value += result
    
    def run(self, total_time):
        start = time.time()
        last_checked = start
        while True:
            now = time.time()
            elapsed = now - start

            best_path = self.get_path_max_visits()
            if elapsed >= total_time and len(best_path) >= (self._max_depth):
                return
            
            if now - last_checked > 10:
                print(f"Been running for {elapsed}s, length of best path {len(best_path)}, depth {self._depth}")
                last_checked = now

            leaf, edges = self.select()
            result = self.expand(leaf)
            if result is not None:
                new_node, edge = result
                edges.append(edge)
                value = self.evaluate(new_node)
                self.backpropagate(new_node, edges, value)
            else:
                # Expand called on fully expanded node
                self.backpropagate(leaf, edges, leaf.get_score())

    def get_map(self):
        return self._map
    
    def _get_or_create_node(self, loc, depth, store=True):
        self._tt_queries += 1
        for (x, y, d), node in self._transposition_table.items():
            if d == depth and self._map.is_same_position((x, y), loc):
                self._tt_hits += 1
                return node
        
        key = (loc[0], loc[1], depth)
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
        leaves = [n for n in self._nodes if n._depth == self._max_depth]
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
        stats = f"Hyperparameters:\n"
        stats += f"c={MonteCarloTree._c} (exploration constant)\n"
        stats += f"k={MonteCarloTree._k} (progressive widening constant)\n"
        stats += f"alpha={MonteCarloTree._alpha} (progressive widening exponent)\n"
        stats += f"beta={MonteCarloTree._beta} (guard distance score)\n"
        stats += f"delta={MonteCarloTree._delta} (distance to visible area score)\n"
        stats += f"theta={MonteCarloTree._theta} (distance to shadow score (for visible nodes)\n"
        stats += f"gamma={MonteCarloTree._gamma} (kernel distance score)\n"
        stats += f"tau={MonteCarloTree._tau} (kernel distance-decay rate)\n"
        stats += f"top_k={MonteCarloTree._top_k} (num of kernels evaluated for score)\n"
        stats += f"eta={MonteCarloTree._eta} (depth bonus)\n"
        stats += f"lambda={MonteCarloTree._lambda} (weight of rollout value in heuristic)\n"
        stats += f"phi={MonteCarloTree._phi} (rollout depth)\n"  
        stats += f"using rollout ratio={MonteCarloTree._rollout_ratio}\n"  
        stats += f"visible node penalty = {MonteCarloTree._visible_node_penalty}\n"
        stats += f"visible edge penalty = {MonteCarloTree._visible_edge_penalty}\n"
        stats += f"using brute force moves = {MonteCarloTree._brute_force}\n"   
        stats += f"max kernel moves = {MonteCarloTree._max_kernel_moves}\n"   

        stats += f"Total nodes in tree: {self._num_nodes}, with max depth {self._depth}, and maximum allowed edges {self._max_edges}\n"
        stats += "Node breadth by depth:\n"
        for depth in sorted(self._breadth):
            stats += f"  Depth {depth}: {self._breadth[depth]} node(s)\n"

        stats += f"Path cache hit ratio {self._map._path_cache_hits / self._map._path_cache_queries}, {self._map._path_cache_hits} hits, {self._map._path_cache_queries} queries\n"
        stats += f"Transposition table hit ratio {self._tt_hits / self._tt_queries}, {self._tt_hits} hits, {self._tt_queries} queries\n"
                                           
        return stats

    class Node:
        def __init__(self, tree : MonteCarloTree, loc : Tuple[float, float], depth : int):
            self._tree = tree
            self._loc = loc
            self._depth = depth
            self._map = tree.get_map()

            self._potential_moves: Optional[List[Tuple[float, float]]] = None
            self._fully_expanded = False
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
                    self._score = MonteCarloTree._visible_node_penalty
                    # add a part of the score of how close it is to shadow though
                    shadow = self._map.get_shadow(self._depth)
                    d = Point(self._loc).distance(shadow)
                    self._score += MonteCarloTree._theta * math.exp(-MonteCarloTree._tau * d)
                    #print(f"Got visible score {self._score}")
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
                kernels = self._map.get_kernels(self._depth, True)
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
                depth_bonus = MonteCarloTree._eta * (self._depth / self._tree._max_depth) ** 2

                # rollout score
                rollout_score = MonteCarloTree._lambda * self._tree.rollout(self)
                    
                self._score = shadow_score + guard_distance_score + kernel_score + depth_bonus + rollout_score
            return self._score
                
        def get_potential_moves(self) -> List[Tuple[int|float, int|float]]:
            if self._potential_moves is None:
                moves = []
                
                if (self._depth == self._tree._max_depth):
                    return moves
                    
                # Compute moves towards kernels
                moves.extend(self.get_moves_towards_kernels())
                # Compute brute force moves
                if (MonteCarloTree._brute_force):
                    moves.extend(self.get_brute_force_moves())

                random.shuffle(moves)
                self._potential_moves = moves

            return self._potential_moves
        
        def get_brute_force_moves(self, num_directions=4) -> List[Tuple[int|float, int|float]]:
            moves = []
            max_step = self._map.get_player_max_step()
            min_step = self._map._prune_tol*2

            x0, y0 = self._loc

            num_samples = 5
            t_values = [
                max_step - (max_step - min_step) * (i / (num_samples - 1))
                for i in range(num_samples)
            ]

            for i in range(num_directions):
                angle = 2 * math.pi * i / num_directions
                dx = math.cos(angle)
                dy = math.sin(angle)

                # Try the farthest point first, then shrink inward
                for t in t_values:
                    candidate = (x0 + t * dx, y0 + t * dy)

                    if self._map.is_valid_position(candidate):
                        moves.append(candidate)
                        break  # stop after first valid distance in this direction

            random.shuffle(moves)
            return moves

        def get_moves_towards_kernels(self) -> List[Tuple[int|float, int|float]]:
            coords = []
            for kernel in self._map.get_kernels(self._depth + 1, True):
                if len(coords) < MonteCarloTree._max_kernel_moves:
                    coords.append(kernel.get_coords())
                else: 
                    break
            return coords

        def __str__(self) -> str:
            avg = self._total_value / self._visits if self._visits > 0 else 0.0
            return (
                f"Node (loc=({self._loc[0]:.2f}, {self._loc[1]:.2f}), "
                f"depth={self._depth}, score={avg:.2f})"
            )


    class Edge: 
        def __init__(self, parent: MonteCarloTree.Node, child: MonteCarloTree.Node, path: LineString, vis_poly: MultiPolygon):
            self._parent = parent
            self._child = child
            self._path = path

            self._visits = 0
            self._total_value = 0.0
            
            self._vis_poly = vis_poly
            self._visible_crossing = self._path.intersects(self._vis_poly)
                
        def ucb_score(self):
            if self._visits == 0:
                return float("inf")
            else:
                exploitation = self._total_value / self._visits
                exploration = MonteCarloTree._c * math.sqrt(
                    math.log(self._parent._visits + 1) / self._visits
                )
                
                if self._visible_crossing:
                    exploration *= MonteCarloTree._visible_edge_penalty
                #print(f"Edge from {self._parent} has ucb score {exploitation + exploration}")
                return exploitation + exploration

        def __str__(self) -> str:
            return f"Edge score={self._total_value/self._visits:.2f}"