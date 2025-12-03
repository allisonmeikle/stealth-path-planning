import os
import math
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
#from descartes import PolygonPatch
from monte_carlo import MonteCarloTree
from map import Map

plt.style.use("classic")

def add_polygon(ax, geom, fc="blue", ec="black", alpha=0.4, **kwargs):
    from shapely.geometry import Polygon, MultiPolygon
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.fill(x, y, fc=fc, ec=ec, alpha=alpha, **kwargs)
        for hole in geom.interiors:
            hx, hy = zip(*hole.coords)
            ax.fill(hx, hy, fc="dimgray", ec=ec)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            add_polygon(ax, g, fc=fc, ec=ec, alpha=alpha, **kwargs)


def plot_shadow_polygon(plot_size, map_poly, obstacles, guard, player, shadow_area, kernels, save_plot = False, file_name = '') :
    os.makedirs('maps', exist_ok=True)
    plt.figure(figsize=plot_size)
    ax = plt.gca()

    # Bounding box (use map bounds with a small buffer)
    minx, miny, maxx, maxy = map_poly.bounds
    bounds = box(minx - 1, miny - 1, maxx + 1, maxy + 1)

    # Combine obstacles
    obstacles_union = unary_union(obstacles)

    # Outside region = bounding box minus map polygon
    outside = bounds.difference(map_poly)

    # Draw outside & obstacles
    add_polygon(ax, outside, fc="dimgray", alpha=1.0, zorder=1)
    add_polygon(ax, map_poly, fc="white", alpha=1.0, zorder=0)
    add_polygon(ax, obstacles_union, fc="dimgray", alpha=2.0)

    # Guard point
    plt.plot(guard.x, guard.y, "r^", markersize=4, label='Guard')

    # Player point
    plt.plot(player.x, player.y, "bo", markersize=4, label='Player')

    # Shadow area
    add_polygon(ax, shadow_area, fc="blue", alpha=0.3, zorder=3)

    kernel_label_added = False

    for k in kernels:
        pt = k.get_point()
        depth = k.get_depth()

        # Only add the label for the first kernel
        if not kernel_label_added:
            ax.plot(pt.x, pt.y, "go", markersize=4, label="Kernel")
            kernel_label_added = True
        else:
            ax.plot(pt.x, pt.y, "go", markersize=4)

        # Annotate with depth slightly above the point
        ax.annotate(
            str(depth),
            (pt.x, pt.y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color="green"
        )

    plt.legend(loc='upper right')

    # Axis setup
    plt.xlim(minx - 1, maxx + 1)
    plt.ylim(miny - 1, maxy + 1)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Map with Guard Visibility (Red) and Shadow (Blue)")
    if (save_plot):
        out_path = os.path.join('maps', file_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
    else: 
        plt.show()

def save_game_state_map(file_name, map_poly, obstacles, guard, player, plot_size = (11,7), out_dir = 'maps'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=plot_size)
    ax = plt.gca()

    # Bounding box (use map bounds with a small buffer)
    minx, miny, maxx, maxy = map_poly.bounds
    bounds = box(minx - 1, miny - 1, maxx + 1, maxy + 1)

    # Combine obstacles
    obstacles_union = unary_union(obstacles)

    # Outside region = bounding box minus map polygon
    outside = bounds.difference(map_poly)

    # Draw outside & obstacles
    add_polygon(ax, outside, fc="dimgray", alpha=1.0)
    add_polygon(ax, obstacles_union, fc="dimgray", alpha=1.0)

    # Guard point
    plt.plot(guard.x, guard.y, "r^", markersize=4, label='Guard')

    # Player point
    plt.plot(player.x, player.y, "bo", markersize=4, label='Player')

    plt.legend(loc='upper right')

    # Axis setup
    plt.xlim(minx - 1, maxx + 1)
    plt.ylim(miny - 1, maxy + 1)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Map with Guard Visibility (Red) and Shadow (Blue)")
    out_path = os.path.join(out_dir, file_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_move(map_poly, obstacles, guard, player, shadow_area, next_point, path, time_step, plot_size=(11, 7), save_plot=False, file_name=''):
    #os.makedirs('maps', exist_ok=True)

    plt.figure(figsize=plot_size)
    ax = plt.gca()

    # Bounding box
    minx, miny, maxx, maxy = map_poly.bounds
    bounds = box(minx - 1, miny - 1, maxx + 1, maxy + 1)

    # Obstacles union
    obstacles_union = unary_union(obstacles)

    # Outside region = bounding box minus map polygon
    outside = bounds.difference(map_poly)

    # Draw outside region (non-walkable)
    add_polygon(ax, outside, fc="dimgray", ec="black", alpha=1.0, zorder=0)

    # Draw map interior (walkable area)
    add_polygon(ax, map_poly, fc="white", ec="black", alpha=1.0, zorder=1)

    # Draw obstacles inside map
    add_polygon(ax, obstacles_union, fc="dimgray", ec="black", alpha=1.0, zorder=2)

    # Shadow polygons
    add_polygon(ax, shadow_area, fc="blue", alpha=0.3, zorder=3)

    # Guard position
    ax.plot(guard.x, guard.y, "r^", markersize=6, label="Guard")

    # Player position
    ax.plot(player.x, player.y, "bo", markersize=6, label="Player")

    # Next target point
    ax.plot(next_point.x, next_point.y, "g*", markersize=10, label="Next Target")

    # Path (LineString)
    if path and not path.is_empty:
        x, y = path.xy
        ax.plot(x, y, "g--", linewidth=2, label="Path")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Axis setup
    plt.xlim(minx - 1, maxx + 1)
    plt.ylim(miny - 1, maxy + 1)
    ax.set_aspect("equal", adjustable="box")
    plt.title(f"Game Step with Guard, Player, Shadow, and Path (time_step = {time_step})")
    
    if save_plot:
        #out_path = os.path.join('maps', file_name)
        plt.savefig(file_name, dpi=150)
        plt.close()
    else: 
        plt.show()

import os
import matplotlib.pyplot as plt
from shapely.geometry import Point

def plot_paths(tree, base_dir="tree_paths"):
    """
    Traverse the MonteCarloTree and save plots of all root-to-leaf paths.
    Each path gets its own directory under base_dir.
    
    Args:
        tree: MonteCarloTree instance
        base_dir: str, where to create directories for each path
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    def dfs(node, path_nodes, path_idx=[0]):
        """
        Depth-first traversal that collects all root-to-leaf paths.
        path_nodes: list of nodes along the current path.
        path_idx: single-element list to keep a counter across recursion.
        """
        path_nodes.append(node)

        if not node.children:  # leaf
            path_idx[0] += 1
            path_dir = os.path.join(base_dir, f"path_{path_idx[0]}")
            os.makedirs(path_dir, exist_ok=True)

            # save a plot for each move in this path
            for i in range(1, len(path_nodes)):
                parent = path_nodes[i - 1]
                child = path_nodes[i]

                file_name = os.path.join(
                    path_dir,
                    f"step_{i:02d}_from_{parent.loc[0]:.2f}_{parent.loc[1]:.2f}"
                    f"_to_{child.loc[0]:.2f}_{child.loc[1]:.2f}.png"
                )

                # call your plotting helper
                plot_move(
                    tree.shapely_map,
                    tree.shapely_obstacles,
                    tree.shapely_guard_positions[child.depth],
                    Point(parent.loc),
                    tree.shadows[child.depth],
                    Point(child.loc),
                    child.path,
                    i,
                    save_plot=True,
                    file_name=file_name
                )
        else:
            # recurse into children
            for child in node.children:
                dfs(child, path_nodes.copy(), path_idx)

    dfs(tree.root, [])

import re
import imageio.v2 as imageio

def natural_key(name):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", name)]

def make_gifs(folder_path, output_name="animation.gif", duration=0.25, loop=True):
    """
    Combine numbered frame_XXX.png images in `folder_path` into a GIF.
    Automatically sorts frames numerically and loops if desired.
    """
    # Collect frame paths
    frames = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".png")
    ]
    frame_files = sorted(frames, key=natural_key)

    # Debug: check order
    print("🧩 Frame order:", [os.path.basename(f) for f in frame_files])

    # --- Read images ---
    images = [imageio.imread(f) for f in frame_files]

    # --- Create output path ---
    output_path = os.path.join(folder_path, output_name)

    # --- Save GIF ---
    imageio.mimsave(
        output_path,
        images,
        duration=duration,
        loop=0 if loop else 1  # 0 = infinite loop
    )

    print(f"✅ Saved GIF → {output_path}")


def plot_vis_poly(map: Map, timestep: int):
    fig, ax = plt.subplots()   # ONE figure, ONE axis
    ax.axis("off")
    guard_pos = map.get_guard_positions(timestep)[0]
    print(guard_pos)
    vis_poly = map.get_visibility_polygon(timestep)
    for pos in guard_pos:
        pass
    

    # Draw outside & obstacles
    add_polygon(ax, map._shapely_boundary, fc="dimgray", alpha=0.5, zorder=1)
    add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", alpha=1.0)
    add_polygon(ax, vis_poly, fc="red", alpha=0.8, zorder=0)

    ax.plot(guard_pos[0], guard_pos[1], "ko")
    ax.text(guard_pos[0] + 0.2, guard_pos[1] + 0.2, "$q$", fontsize=14)

    plt.savefig("visibility_polygon_example.png", dpi=300)

def plot_path(tree, path, save_dir, label):
    os.makedirs(save_dir, exist_ok=True)
    
    path_pts = []
    in_shadow = 0
    for i, edge in enumerate(path):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")
        ax.axis("off")

        map = tree.get_map()
        node = edge._parent
        guard_positions = map.get_guard_positions(node._depth)
        player_pos = node.get_loc()
        if not tree._map.is_visible(player_pos, node._depth):
            in_shadow += 1

        # draw layers
        add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0)
        add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", ec="black", alpha=1.0)
        add_polygon(ax, map.get_shadow(node._depth), fc="blue", alpha=0.25)

        # draw entities
        for gp in guard_positions:
            ax.plot(gp[0], gp[1], "r^", markersize=4)
        ax.plot(player_pos[0], player_pos[1], "bo", markersize=4)

        # draw kernels
        kernels = tree._map.get_kernels(i)
        for k in kernels:
            kx, ky = k.get_coords()
            ax.plot(kx, ky, "ro", markersize=4)
            ax.text(kx+0.05, ky+0.05, f"{k.get_depth()}", fontsize=6)

        # draw path so far
        path_pts.extend(list(edge._path.coords))
        xs, ys = zip(*path_pts)
        ax.plot(xs, ys, "g--", lw=2)

        ax.set_title(f"{label}: t={node._depth}, Q/N={node._total_value/node._visits:.2f}, N={node._visits}")

        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

    make_gifs(save_dir)

    return in_shadow, len(path)


def output_results(tree: MonteCarloTree, run_time: str, save_dir="plots/best_path_gif", duration=0.25):
    os.makedirs(save_dir, exist_ok=True)
    p_visits   = os.path.join(save_dir, "best_path_max_visits")
    p_score    = os.path.join(save_dir, "best_path_max_score")
    p_back     = os.path.join(save_dir, "best_path_backtrack")

    # extract all 3 paths
    path_visits = tree.get_path_max_visits()
    path_score  = tree.get_path_max_score()
    path_back   = tree.get_path_backtracked()

    # plot each
    shadow_ct1, total1 = plot_path(tree, path_visits, p_visits, "Max-visits")
    shadow_ct2, total2 = plot_path(tree, path_score,  p_score,  "Max-score")
    shadow_ct3, total3 = plot_path(tree, path_back,   p_back,   "Backtracked full-depth")

    # write stats
    stats = tree.get_stats()
    stats += f"Max-visits path gives {shadow_ct1} of {total1} ({shadow_ct1/total1*100:.2f}%) steps in shadow\n"
    stats += f"Max-score path gives {shadow_ct2} of {total2} ({shadow_ct2/total2*100:.2f}%) steps in shadow\n"
    stats += f"Backtracked path gives {shadow_ct3} of {total3} ({shadow_ct3/total3*100:.2f}%) steps in shadow\n"
    stats += f"Took {run_time} to find the best path\n"

    with open(os.path.join(save_dir, "stats.txt"), 'w') as f:
        f.write(stats)

def plot_guard_path(map, save_dir="plots/guard_path"):
    os.makedirs(save_dir, exist_ok=True)

    frames = []

    num_timesteps = map.get_num_timesteps()

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # Draw map
        add_polygon(ax, map._shapely_boundary, fc="white", ec="black", alpha=1.0)
        add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", ec="black", alpha=1.0)

        # Draw ALL guards at this timestep
        for i, guard in enumerate(map._guards):
            gx, gy = guard.get_path()[t]
            ax.plot(gx, gy, "r^", markersize=8, label=f"Guard {i}")

        ax.set_title(f"Guard positions at timestep {t}", fontsize=12)

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

        frame_path = os.path.join(save_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(frame_path)

    # Make GIF
    make_gifs(save_dir)
    print(f"Guard path GIF saved to {save_dir}")

def plot_shadow_and_kernels(map, save_dir="plots/shadow_kernels"):
    os.makedirs(save_dir, exist_ok=True)

    num_timesteps = map.get_num_timesteps()
    boundary = map._shapely_boundary
    obstacles = unary_union(map._shapely_obstacles)

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", "box")
        ax.axis("off")

        # -------------------------------------------------------
        # Draw Map
        # -------------------------------------------------------
        add_polygon(ax, boundary, fc="white", ec="black", alpha=1.0, zorder=1)
        add_polygon(ax, obstacles, fc="dimgray", ec="black", alpha=1.0, zorder=2)

        # -------------------------------------------------------
        # Shadows
        # -------------------------------------------------------
        shadow = map._shadows[t]
        shadow_w_obs = map._shadows_w_obs[t]

        add_polygon(ax, shadow, fc="blue", alpha=0.30, zorder=3, label="Shadow")
        add_polygon(ax, shadow_w_obs, fc="purple", alpha=0.25, zorder=4, label="Shadow w/ Obstacles")

        # -------------------------------------------------------
        # Kernels
        # -------------------------------------------------------
        kernels_no_obs = map._kernels[t] if map._kernels else map.get_kernels(t)
        kernels_w_obs = map._kernels_w_obs[t] if map._kernels_w_obs else map.get_kernels(t)

        # Green = kernel from shadow
        for k in kernels_no_obs:
            kx, ky = k.get_coords()
            ax.plot(kx, ky, "go", markersize=5, zorder=5, label="Kernel (no obs)")

        # Orange = kernel from shadow including obstacles
        for k in kernels_w_obs:
            kx, ky = k.get_coords()
            ax.plot(kx, ky, "o", color="orange", markersize=5, zorder=6, label="Kernel (w obs)")

        # -------------------------------------------------------
        # Guard positions
        # -------------------------------------------------------
        for i, guard in enumerate(map._guards):
            gx, gy = guard.get_path()[t]
            ax.plot(gx, gy, "r^", markersize=8, zorder=7, label=f"Guard {i}")

        # -------------------------------------------------------
        # Final formatting
        # -------------------------------------------------------
        ax.set_title(f"Shadows + Kernels at timestep {t}", fontsize=12)

        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)

        out_path = os.path.join(save_dir, f"timestep_{t:03d}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")