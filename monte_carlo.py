from __future__ import annotations
from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from extremitypathfinder import PolygonEnvironment

from kernel import find_kernels, Kernel
from polygon_helpers import *
from plot_helper import *

import numpy as np
import math

class MonteCarloTree: 
    c = np.sqrt(2)

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
        self.root = MonteCarloTree.Node(0, player_start_pos)

        # Initialize PolygonEnvironment for computing shortest paths btw points
        self.env = PolygonEnvironment()
        self.env.store(map, obstacles, True)
        self.env.prepare()
        
    def select(self) -> MonteCarloTree.Node:
        current = self.root
        while True:
            print("Current node in selection loop: ", current)
            num_reachable_kernels = len(get_reachable_kernels(self.env, self.max_step, current.loc, self.kernels[current.depth]))
            if (num_reachable_kernels == 0):
                return current
            elif num_reachable_kernels > len(current.children):
                return self.expand(current)
            else: 
                current = max(current.children, key=lambda child: child.ucb_score())
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        explored_kernels = {child.kernel for child in node.children}
        for kernel, path in get_reachable_kernels(self.env, self.max_step, node.loc, self.kernels[node.depth]):
            if (kernel not in explored_kernels):
                print(kernel, path)
        
        return MonteCarloTree.Node(node.depth + 1, kernel._point, kernel, path, node)


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
        i = 1
        for kernel in self.kernels[0]:
            target = kernel.get_coords()
            path, length = self.env.find_shortest_path(self.root.loc, target)
            coords = [(p[0], p[1]) for p in path]
            line = LineString(coords)
            plot_move(self.shapely_map, self.shapely_obstacles, self.shapely_guard_positions[0], Point(self.root.loc), self.shadows[0], Point(target), line, save_plot=True, file_name=f'map_to_kernel_{i}')
            i += 1

        '''
        selected = self.select()
        result = self.evaluate(selected)
        self.backpropagate(selected, result)
        return
        '''

    class Node:
        def __init__(
                self, 
                depth : int, 
                loc : Tuple[float, float], 
                kernel : Optional[Kernel] = None,
                path : Optional[LineString] = None,
                parent : Optional[MonteCarloTree.Node] = None,
                children : List[MonteCarloTree.Node] = [],
            ):
            self.depth = depth
            self.loc = loc
            self.kernel = kernel
            self.path = path
            self.parent = parent
            self.children = children
            self.score = 0
            self.visits = 0

        def ucb_score(self):
            exploitation = self.score / self.visits
            exploration = MonteCarloTree.c * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
            return exploitation + exploration
        
        def __str__(self) -> str:
            return f"Node (loc=({self.loc[0]:.2f}, {self.loc[1]:.2f}), depth={self.depth})"