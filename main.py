from monte_carlo import MonteCarloTree
from characters import *
from map import *

# Player parameters
player = Player(0.5, 2.0, (1.5, 8.0))

# Guard parameters
guard_positions = [
    (16.0, 5.0),
    (16.0, 3.5),
    (15.515, 1.560),
    (13.516, 1.5),
    (11.516, 1.5)
]
guard = Guard(0.5, 2.0, guard_positions)

# Map parameters
boundary = [
    (3.0, 0.0), (5.0, 0.0), (5.0, 1.0), (18.0, 1.0),
    (18.0, 10.0), (19.0, 10.0), (19.0, 11.0),
    (15.0, 11.0), (15.0, 10.0), (14.0, 10.0), (14.0, 11.0), (0.0, 11.0),
    (0.0, 7.0), (3.0, 7.0), (3.0, 9.0), (6.0, 9.0), (6.0, 3.0), (3.0, 3.0)
]

obstacles = [
    [(7.0, 4.0), (7.0, 6.0), (9.0, 6.0), (9.0, 4.0)],
    [(8.0, 7.0), (8.0, 9.0), (9.0, 9.0), (9.0, 8.0), (10.0, 8.0), (10.0, 9.0), (11.0, 9.0), (11.0, 7.0)],
    [(14.0, 6.0), (14.0, 8.0), (16.0, 8.0), (16.0, 6.0)],
    [(11.0, 2.0), (11.0, 4.0), (15.0, 4.0), (15.0, 2.0), (14.0, 2.0), (14.0, 3.0), (12.0, 3.0), (12.0, 2.0)]
]
map = Map((20, 12), boundary, obstacles, [guard], player)
map.compute_kernels(0.05)

# Running MCTS
monte_carlo_tree = MonteCarloTree(map, player, guard)
result = monte_carlo_tree.run()