from __future__ import annotations
from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from extremitypathfinder import PolygonEnvironment

from kernel import find_kernels, Kernel
from polygon_helpers import *
from plot_helper import *

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
                # Optimal path to a leaf has been found
                return None
            
            num_potential_moves = len(current.get_potential_moves())
            if (num_potential_moves == 0):
                raise RuntimeError("During selection, found a node with no potential moves that is not a leaf.")
            if num_potential_moves > len(current.children):
                return self.expand(current)
            else: 
                current = max(current.children, key=lambda child: child.ucb_score())
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        '''
        explored_kernels = {child.kernel for child in node.children}
        for kernel, path in get_reachable_kernels(self.env, self.max_step, node.loc, self.kernels[node.depth]):
            if (kernel not in explored_kernels):
                print(kernel, path)
        '''
        return MonteCarloTree.Node(self, node.depth + 1, (0, 0), False)


    def evaluate(self, node : MonteCarloTree.Node):
        '''
        give a heuristic estimate here of how good this location would be to go to
        
        - is the location in shadow
        - how many reachable kernels are there from this point
        - how far away is this point from the guard 
        - what is the depth of the kernel? (this would only apply if i make it so that each node is a kernel)
        '''
        plot_move(self.map, self.obstacles, self.guard_positions[node.depth-1], node.parent.loc, self.shadows[node.depth-1], node.loc, node.path)
        return
    
    def backpropagate(self, node : MonteCarloTree.Node, result):
        return
    
    def run(self):
        selected = self.select()
        if (selected is None):
            print("Found optimal path!")
            return True
        
        #result = self.evaluate(selected)
        #self.backpropagate(selected, result)
        return False

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
                children : List[MonteCarloTree.Node] = [],
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
            exploitation = self.score / self.visits
            exploration = MonteCarloTree.c * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
            return exploitation + exploration
        
        def __str__(self) -> str:
            return f"Node (loc=({self.loc[0]:.2f}, {self.loc[1]:.2f}), depth={self.depth})"
        
        def get_potential_moves(self) -> List[Tuple[LineString, Tuple[float, float]]]:
            """
            Compute all possible moves from the current node.

            - If the shortest path to a kernel is <= max_step, add the kernel location.
            - If the path is longer, add the point along the path at max_step distance.
            """
            print("Called get potential moves")
            if (self.potential_moves is None):
                print("computing potential moves")
                moves = []
                if (self.depth == len(self.tree.shadows)-1):
                    return moves
                
                for kernel in self.tree.kernels[self.depth]: 
                    target = kernel.get_coords()
                    print(f"Finding path to target {target}")
                    path, length = self.tree.env.find_shortest_path(self.loc, target)
                    if not path or length is None:
                        continue

                    line = LineString(path)
                    if length <= self.tree.max_step:
                        # Entire kernel is reachable
                        moves.append((line, target))
                    else:
                        # Take a point along the path at max_step distance
                        pt = line.interpolate(self.tree.max_step)
                        moves.append((line, (pt.x, pt.y)))

                self.potential_moves = moves
                i = 1
                for move in self.potential_moves:
                    plot_move(self.tree.shapely_map, self.tree.shapely_obstacles, self.tree.shapely_guard_positions[self.depth], Point(self.loc), self.tree.shadows[self.depth], Point(move[1]), move[0], save_plot=True, file_name=f"map_from_root_to_kernel_{i}")
                    i += 1

            return self.potential_moves