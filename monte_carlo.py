from __future__ import annotations
from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString

from kernel import find_kernels, Kernel
from polygon_helpers import *
from plot_helper import *

import pyvisgraph as vg
import numpy as np
import math

class MonteCarloTree: 
    c = np.sqrt(2)

    def __init__(self, map, obstacles, player_start_pos, guard_positions, max_step):
        self.map = map
        self.obstacles = obstacles
        self.guard_positions = guard_positions
        self.max_step = max_step
        self.visibility_graph = build_visibility_graph(map, obstacles)
        self.shadows = compute_shadows(map, obstacles, guard_positions)
        self.kernels = compute_kernels(self.shadows, 0.01)
        self.root = MonteCarloTree.Node(0, player_start_pos)

    def select(self) -> MonteCarloTree.Node:
        current = self.root
        while True:
            print("Current node in selection loop: ", current)
            num_reachable_kernels = len(get_reachable_kernels(self.visibility_graph, self.max_step, current.loc, self.kernels[current.depth]))
            if (num_reachable_kernels == 0):
                return current
            elif num_reachable_kernels > len(current.children):
                return self.expand(current)
            else: 
                current = max(current.children, key=lambda child: child.ucb_score())
        return self.root
    
    def expand(self, node : MonteCarloTree.Node) -> MonteCarloTree.Node:
        explored_kernels = {child.kernel for child in node.children}
        for kernel, path in get_reachable_kernels(self.visibility_graph, self.max_step, node.loc, self.kernels[node.depth]):
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
        selected = self.select()
        result = self.evaluate(selected)
        self.backpropagate(selected, result)
        return

    class Node:
        def __init__(
                self, 
                depth : int, 
                loc : Point, 
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
            return f"Node (loc=({self.loc.x:.2f}, {self.loc.y:.2f}), depth={self.depth})"