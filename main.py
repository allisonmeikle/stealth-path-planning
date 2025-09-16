from shapely.geometry import Polygon, Point
from monte_carlo import MonteCarloTree

# Game parameters
max_step = 2
player_start_pos = Point(1.5, 8)

map = Polygon([
    # Bottom boundary
    (3, 0), (5, 0), (5, 1), (18, 1), 
    # Right boundary
    (18, 10), (19, 10), (19, 11),
    # Top boundary
    (15, 11), (15, 10), (14, 10), (14, 11), (0, 11),
    # Left boundary
    (0, 7), (3, 7), (3, 9), (6, 9), (6, 3), (3, 3)
    ])

obstacles = [
    Polygon([(7,4), (7, 6), (9, 6), (9,4)]),
    Polygon([(8, 7), (8, 9), (9, 9), (9, 8), (10, 8), (10, 9), (11, 9), (11, 7)]),
    Polygon([(14, 6), (14, 8), (16, 8), (16, 6)]),
    Polygon([(11, 2), (11, 4), (15, 4), (15, 2), (14, 2), (14, 3), (12, 3), (12, 2)]),
]

guard_positions = [
    Point(16, 5),
    Point(16, 3.5),
    Point(15.515, 1.560),
    Point(13.516, 1.5),
    Point(11.516, 1.5)
]



from polygon_helpers import *
from plot_helper import *
import pyvisgraph as vg

visibility_graph = build_visibility_graph(map, obstacles)
shadows = compute_shadows(map, obstacles, guard_positions)
kernels = compute_kernels(shadows, 0.01)

start = vg.Point(player_start_pos.x, player_start_pos.y)
for i in range (len(kernels[0])):
    print(i)
    kp = kernels[0][i].get_point()
    path = visibility_graph.shortest_path(start, vg.Point(kp.x, kp.y))
    if not path:
        continue
    # Convert to shapely LineString
    coords = [(p.x, p.y) for p in path]
    line = LineString(coords)

    plot_move(map, obstacles, guard_positions[0], player_start_pos, shadows[0], kp, line, save_plot=True, file_name=f'map_{i}')

'''
monte_carlo_tree = MonteCarloTree(map, obstacles, player_start_pos, guard_positions, max_step)
monte_carlo_tree.run()
'''