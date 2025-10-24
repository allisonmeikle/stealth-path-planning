import math
from monte_carlo import MonteCarloTree
from characters import *
from map import *
from plot_helper import *


# Player parameters
player = Player(0.1, 0.5, (1.5, 8.0))

# Guard parameters
'''
guard_positions = [
    (16.0, 5.0),
    (16.0, 3.5),
    (15.515, 1.560),
    (13.516, 1.5),
    (11.516, 1.5)
]
'''
# center of the obstacle
cx, cy = 5, 5
radius = 2.0        # 2 units away from center
step_size = 0.5
num_points = int((2 * math.pi * radius) / step_size)

guard_positions = [
    (cx + radius * math.cos(theta), cy + radius * math.sin(theta))
    for theta in [i * (2 * math.pi / num_points) for i in range(num_points)]
]
guard = Guard(0.5, 2.0, guard_positions)

# Map parameters
'''
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
'''
boundary = [(0,0), (10,0), (10,10), (0,10)]  # CCW (outer boundary)
obstacles = [[(4,4), (4,6), (6,6), (6,4)]]   # CW (inner obstacle)

#map = Map((20, 12), boundary, obstacles, [guard], player)
map = Map((10, 10), boundary, obstacles, [guard], player)
map.compute_kernels(0.01)
#map.visualize_kernel_evolution(5)
#map.save_map_states()


# Running MCTS
monte_carlo_tree = MonteCarloTree(map)
result = monte_carlo_tree.run()
#MonteCarloTree.traverse_and_plot(monte_carlo_tree.root)
#MonteCarloTree.Node.plot_move(monte_carlo_tree.root, plot_size=(11, 7))

def visualize_best_path(tree: MonteCarloTree, save_dir="plots/best_path_gif", duration=0.25):
    os.makedirs(save_dir, exist_ok=True)
    best_leaf = tree.get_best_leaf()
    path_nodes = get_path_to_root(best_leaf)

    frames = []
    for i, node in enumerate(path_nodes):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", "box")
        ax.axis("off")

        map = node._map
        guard_pos = map.get_guard_positions(node._depth)[0] if callable(map.get_guard_positions) else map._guard.get_path()[node._depth]
        player_pos = node.get_loc()

        # --- Draw map layers ---
        add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0)
        add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", ec="black", alpha=1.0)
        add_polygon(ax, map._shadows[node._depth], fc="blue", alpha=0.25)

        # --- Draw positions ---
        ax.plot(guard_pos[0], guard_pos[1], "r^", markersize=8, label="Guard")
        ax.plot(player_pos[0], player_pos[1], "bo", markersize=8, label="Player")

        # --- Draw path so far ---
        if i > 0:
            prev_pts = [n.get_loc() for n in path_nodes[:i+1]]
            xs, ys = zip(*prev_pts)
            ax.plot(xs, ys, "g--", lw=2, alpha=0.7, label="Path")

        ax.set_title(f"Timestep {node._depth} | Score {node._score/node._num_visits:.2f}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)

        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(frame_path)

def get_path_to_root(node: "MonteCarloTree.Node") -> list["MonteCarloTree.Node"]:
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = cur._parent
    path.reverse()
    return path

# Combine into GIF
visualize_best_path(monte_carlo_tree, duration=0.3)
make_gifs('plots/best_path_gif')