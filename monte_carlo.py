from __future__ import annotations
from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from extremitypathfinder import PolygonEnvironment

from kernel import find_kernels, Kernel
from polygon_helpers import *
from plot_helper import *
from helpers import *

import math

class MonteCarloTree: 
    c = math.sqrt(2)

    '''
    Initializes a MonteCarloTree to run MCTS on.

    Args:
        map: The outer-boundary of the map, defined as a list of (x, y) coordinates in counter-clockwise order.
        obstacles: A list of obstacles inside of the map. Each obstacle is a list of (x, y) coordinates in clockwise order.
        player_start_pos: (x, y) coordinate of where the player starts in the map.
        guard_positions: A list of the guard's positions as (x, y) coordinates at each step throughout the game. 
        max_step: The maximum distance a player can move in one step. 

    '''
    def __init__(self, map: List[Tuple[float, float]], obstacles: List[List[Tuple[float, float]]], player_start_pos: Tuple[float, float], guard_positions: List[Tuple[float, float]], max_step: float):
        self.shapely_map = Polygon(map)
        self.shapely_obstacles = []
        for obstacle in obstacles:
            self.shapely_obstacles.append(Polygon(obstacle))

        self.shapely_guard_positions = []
        for position in guard_positions:
            self.shapely_guard_positions.append(Point(position))

        self.max_step = max_step
        self.shadows = compute_shadows(self.shapely_map, self.shapely_obstacles, self.shapely_guard_positions)
        self.kernels = compute_kernels(self.shadows, 0.01)
        self.root = MonteCarloTree.Node(self, 0, player_start_pos, False)

        # Initialize PolygonEnvironment for computing shortest paths btw points
        self.env = PolygonEnvironment()
        self.env.store(map, obstacles, True)
        self.env.prepare()
        
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
                file_name = f"map_from_{node.get_loc()[0]:.2f}_{node.get_loc()[1]:.2f}_to_{loc[0]:.2f}_{loc[1]:.2f}.png"
                plot_move(self.shapely_map, self.shapely_obstacles, self.shapely_guard_positions[new_child.depth], Point(node.get_loc()), self.shadows[new_child.depth], Point(loc), path, new_child.depth, save_plot=True, file_name=file_name)
                return new_child
        
        raise RuntimeError("Expand was called on a fully-expanded node")

    def evaluate(self, node : MonteCarloTree.Node):
        '''
        give a heuristic estimate here of how good this location would be to go to
        
        - is the location in shadow
        - how many reachable kernels are there from this point
        - how far away is this point from the guard 
        - what is the depth of the kernel? (this would only apply if i make it so that each node is a kernel)
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
            """
            Compute all possible moves from the current node.

            - If the shortest path to a kernel is <= max_step, add the kernel location.
            - If the path is longer, add the point along the path at max_step distance.
            - Prune moves that are within `prune_tol` distance of one another.
            """

            if (self.potential_moves is None):
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
                self.potential_moves = moves

            return self.potential_moves