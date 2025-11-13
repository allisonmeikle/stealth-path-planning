import argparse
import json
import time
from datetime import datetime
import os.path
from monte_carlo import MonteCarloTree
from characters import *
from map import *
from plot_helper import *

CUR_DIR = os.path.dirname(__file__)

def is_valid_position(pos):
    return isinstance(pos, list) and len(pos) == 2 and all(isinstance(x, (int, float)) for x in pos)

def is_valid_guard(guard):
    if not isinstance(guard, dict):
        return False
    
    radius = guard.get("radius")
    if not radius or not isinstance(radius, float):
        return False
    
    positions = guard.get("positions")
    if not positions or not isinstance(positions, list) or not all(is_valid_position(x) for x in positions):
        return False
    
    return True

def load_level_info(file_name):
    full_path = os.path.join(CUR_DIR, 'maps', file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Level file not found: {full_path}")
    try:
        with open(full_path, "r") as f:
            level = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {file_name}: {e}")
    
    # Player info
    player_start = level.get("player_start")
    if player_start and not is_valid_position(player_start):
        raise ValueError(f"Player start parameter {player_start} is invalid")
    
    player_radius = level.get("player_radius")
    if not player_radius or not isinstance(player_radius, float):
        raise ValueError(f"Player radius is invalid, must be of type float")
    
    player_speed = level.get("player_speed")
    if not player_speed or not isinstance(player_speed, float):
        raise ValueError(f"Player speed is invalid, must be of type float")
    
    player = Player(player_radius, player_speed, player_start)

    # Guard(s) info 
    guards = level.get("guards")
    if not guards or not isinstance(guards, list) or not all(is_valid_guard(guard) for guard in guards):
        raise ValueError("Guard(s) parameter missing or invalid")
    
    guard_objs = []
    for guard in guards:
        guard_objs.append(Guard(guard.get("radius"), guard.get("speed"), guard.get("positions")))

    # Map info
    grid_size = level.get("grid_size")
    if not grid_size or not is_valid_position(grid_size):
        raise ValueError("Grid size parameter missing or invalid")

    boundary = level.get("boundary")
    if not boundary or not all(is_valid_position(x) for x in boundary):
        raise ValueError("Map boundary parameter missing or invalid")

    obstacles = level.get("obstacles")
    print(obstacles)
    print((not obstacles), (not isinstance(obstacles, list)), not all(isinstance(poly, list) and all(is_valid_position(p) for p in poly) for poly in obstacles))
    if not isinstance(obstacles, list) or not all(isinstance(poly, list) and all(is_valid_position(p) for p in poly) for poly in obstacles):
        raise ValueError("Map obstacles parameter missing or invalid")
    
    map = Map(grid_size, boundary, obstacles, guard_objs, player)

    return player, guard_objs, map

def output_results(tree: MonteCarloTree, run_time: str, save_dir="plots/best_path_gif", duration=0.25):
    os.makedirs(save_dir, exist_ok=True)
    stats = tree.get_stats()
    path = tree.find_best_path_to_max_depth(tree._root)

    with open(os.path.join(save_dir, "stats.txt"), 'w') as f:
        stats += "Took " + run_time + " to find the best path"
        f.write(stats)

    frames = []
    path_pts = []
    for i, edge in enumerate(path):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", "box")
        ax.axis("off")

        map = tree.get_map()
        node = edge._parent
        guard_positions = map.get_guard_positions(node._depth)
        player_pos = node.get_loc()

        # --- Draw map layers ---
        add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0)
        add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", ec="black", alpha=1.0)
        add_polygon(ax, map.get_shadow(node._depth), fc="blue", alpha=0.25)

        # --- Draw positions ---
        for guard_pos in guard_positions:
            ax.plot(guard_pos[0], guard_pos[1], "r^", markersize=4, label="Guard")
        ax.plot(player_pos[0], player_pos[1], "bo", markersize=4, label="Player")

        # --- Draw path so far ---
        path_pts.extend(list(edge._path.coords))
        xs, ys = zip(*path_pts)
        ax.plot(xs, ys, "g--", lw=2, alpha=0.7, label="Path")

        ax.set_title(f"Timestep {node._depth} | Score {node._total_value/node._visits:.2f}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)

        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(frame_path)

def plot_guard_path(map, save_dir="plots/guard_path"):
    os.makedirs(save_dir, exist_ok=True)

    frames = []

    guard = map._guards[0]   # assume one guard for now
    guard_positions = guard.get_path()

    for t, pos in enumerate(guard_positions):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # Draw map
        add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0)
        add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", ec="black", alpha=1.0)

        # Draw guard
        ax.plot(pos[0], pos[1], "r^", markersize=8, label="Guard")

        ax.set_title(f"Guard position at timestep {t}", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)

        frame_path = os.path.join(save_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(frame_path)

    # Make GIF
    make_gifs(save_dir)
    print(f"Guard path GIF saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level", help="Name of the JSON file in the ./maps directory")
    args = parser.parse_args()

    player, guards, map = load_level_info(args.level)
    '''
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(CUR_DIR, "results", "guard_paths", f"guards_{timestamp}")
    plot_guard_path(map, save_dir=save_dir)
    return
    '''
    #map.plot_shadow_comparison('plots/shadow_comparison1')
    monte_carlo_tree = MonteCarloTree(map)
    start_time = time.time()
    monte_carlo_tree.run(total_time = 600)
    end_time = time.time()
    print(f"Monte Carlo run took {(end_time - start_time):.3f} seconds")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = os.path.join(CUR_DIR, "results", "nov-13", f"results_{timestamp}")
    run_time = f"{(end_time - start_time):.3f}"
    output_results(monte_carlo_tree, run_time, save_dir=dirname, duration=0.3)
    make_gifs(dirname)    

if __name__ == "__main__":
    main()
