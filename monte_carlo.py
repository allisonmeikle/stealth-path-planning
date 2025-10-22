from __future__ import annotations

import math

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

    def get_max_step(self):
        return self._player.get_max_step()
    
    def get_shortest_path(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        return self._map.get_shortest_path(pt1, pt2)
        
    def select(self) -> Optional[MonteCarloTree.Node]:
        current = self.root
        while True:
            print("Current node in selection loop: ", current)

            if (current.depth >= (len(self.shapely_guard_positions)-1)):
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
        '''
        give a heuristic estimate here of how good this location would be to go to
        
        - is the location in shadow
        - how many reachable kernels are there from this point
        - how far away is this point from the guard 
        - what is the depth of the kernel? (this would only apply if i make it so that each node is a kernel)

        find distance to nearest shadow
        find distance to nearest kernel
        '''
        return 1
    
    def backpropagate(self, node : MonteCarloTree.Node, result):
        current = node
        while current is not None:
            current.visits += 1
            current.score += result
            print(f"Backpropagating to {current}")
            current = current.parent
    
    def run(self):
        while True:
            selected = self.select()
            if (selected is None):
                print("Found optimal path!")
                plot_paths(self)
                return
            
            result = self.evaluate(selected)
            self.backpropagate(selected, result)

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

            self.score = 0
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
                num_directions = 8
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