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
    if not positions or not isinstance(positions, list) or not len(positions) > 1 or not all(is_valid_position(x) for x in positions):
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
    if not isinstance(obstacles, list) or not all(isinstance(poly, list) and all(is_valid_position(p) for p in poly) for poly in obstacles):
        raise ValueError("Map obstacles parameter missing or invalid")
    
    map = Map(file_name, grid_size, boundary, obstacles, guard_objs, player)
    return map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level", help="Name of the JSON file in the ./maps directory")
    args = parser.parse_args()

    map = load_level_info(args.level)
    monte_carlo_tree = MonteCarloTree(map, max_edges=100)
    start_time = time.time()
    monte_carlo_tree.run(60)
    end_time = time.time()
    print(f"Monte Carlo run took {(end_time - start_time):.3f} seconds")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = os.path.join(CUR_DIR, "results", "dec-12", f"results_{timestamp}")
    run_time = f"{(end_time - start_time):.3f}"
    output_results(monte_carlo_tree, run_time, save_dir=dirname, duration=0.3, save=True)
    os.system('say "Monte Carlo run complete"')

if __name__ == "__main__":
    main()
   

