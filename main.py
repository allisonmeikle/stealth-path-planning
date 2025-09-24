from monte_carlo import MonteCarloTree

# Game parameters
max_step = 2.0
player_start_pos = (1.5, 8.0)

map = [
    # Bottom boundary
    (3.0, 0.0), (5.0, 0.0), (5.0, 1.0), (18.0, 1.0),
    # Right boundary
    (18.0, 10.0), (19.0, 10.0), (19.0, 11.0),
    # Top boundary
    (15.0, 11.0), (15.0, 10.0), (14.0, 10.0), (14.0, 11.0), (0.0, 11.0),
    # Left boundary
    (0.0, 7.0), (3.0, 7.0), (3.0, 9.0), (6.0, 9.0), (6.0, 3.0), (3.0, 3.0)
]

obstacles = [
    [(7.0, 4.0), (7.0, 6.0), (9.0, 6.0), (9.0, 4.0)],
    [(8.0, 7.0), (8.0, 9.0), (9.0, 9.0), (9.0, 8.0), (10.0, 8.0), (10.0, 9.0), (11.0, 9.0), (11.0, 7.0)],
    [(14.0, 6.0), (14.0, 8.0), (16.0, 8.0), (16.0, 6.0)],
    [(11.0, 2.0), (11.0, 4.0), (15.0, 4.0), (15.0, 2.0), (14.0, 2.0), (14.0, 3.0), (12.0, 3.0), (12.0, 2.0)]
]

guard_positions = [
    (16.0, 5.0),
    (16.0, 3.5),
    (15.515, 1.560),
    (13.516, 1.5),
    (11.516, 1.5)
]

# Running MCTS
monte_carlo_tree = MonteCarloTree(map, obstacles, player_start_pos, guard_positions, max_step)
result = monte_carlo_tree.run()
'''
while (not result):
    result = monte_carlo_tree.run()
'''