import os
import csv
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
#from descartes import PolygonPatch
from monte_carlo import MonteCarloGraph
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

def plot_shadow_poly(map: Map, timestep: int):
    fig, ax = plt.subplots()   # ONE figure, ONE axis
    ax.axis("off")
    guard_pos = map.get_guard_positions(timestep)[0]
    vis_poly = map.get_visibility_polygon(timestep)
    shadow_poly = map.get_shadow(timestep)

    # Draw outside & obstacles
    add_polygon(ax, map._shapely_boundary, fc="dimgray", alpha=0.5, zorder=1)
    add_polygon(ax, unary_union(map._shapely_obstacles), fc="dimgray", alpha=1.0)
    add_polygon(ax, vis_poly, fc="red", alpha=0.8, zorder=0)
    add_polygon(ax, shadow_poly, fc="blue", alpha=0.8, zorder=0)

    ax.plot(guard_pos[0], guard_pos[1], "ko")
    ax.text(guard_pos[0] + 0.2, guard_pos[1] + 0.2, "$q$", fontsize=14)

    plt.savefig("visibility_polygon_example.png", dpi=300)

from matplotlib.lines import Line2D


def plot_static_colored_paths_with_spotted(tree, path, out_path, show=False):
    """
    Given a MonteCarloGraph `tree` and a best `path` (list of edges),
    draw a static colour-coded plot with:

      - Player path (colour changes with timestep)
      - All guard paths (same colour scheme)
      - Red X markers where the player is visible (spotted)
      - Legend entries for player start/end and guard start/end
    """

    if not path:
        print("plot_static_colored_paths_with_spotted: empty path, skipping.")
        return

    map_obj = tree.get_map()

    # --- Reconstruct player positions and timesteps from the path ---
    player_positions = []
    timesteps = []

    for edge in path:
        node = edge._parent
        player_positions.append(node.get_loc())
        timesteps.append(node._depth)

    final_node = path[-1]._child
    player_positions.append(final_node.get_loc())
    timesteps.append(final_node._depth)

    combined = sorted(zip(timesteps, player_positions), key=lambda x: x[0])
    timesteps, player_positions = zip(*combined)
    timesteps = list(timesteps)
    player_positions = list(player_positions)

    t_min, t_max = min(timesteps), max(timesteps)
    if t_max == t_min:
        t_max = t_min + 1

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Best Path (colour-coded by timestep)", pad=28)
    ax.axis("off")

    # --- Draw map geometry ---
    add_polygon(ax, map_obj._shapely_boundary, fc="white", ec="black", alpha=1.0)

    if map_obj._shapely_obstacles:
        add_polygon(
            ax,
            unary_union(map_obj._shapely_obstacles),
            fc="dimgray",
            ec="black",
            alpha=1.0,
        )

    # --- Colormap & normalizer over timesteps ---
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=t_min, vmax=t_max)

    # --- Guard paths (all guards) ---
    for guard in map_obj._guards:
        guard_path = guard.get_path()
        T_guard = len(guard_path)

        # Colour-coded segments (same cmap/norm as player)
        for i in range(len(timesteps) - 1):
            t0 = timesteps[i]
            if t0 + 1 >= T_guard:
                continue
            (gx0, gy0) = guard_path[t0]
            (gx1, gy1) = guard_path[t0 + 1]
            ax.plot(
                [gx0, gx1],
                [gy0, gy1],
                color=cmap(norm(t0)),
                linewidth=1.5,
                alpha=0.8,
            )

        # Mark first/last positions for this guard
        g_start_t = timesteps[0]
        g_end_t = min(timesteps[-1], T_guard - 1)
        sx, sy = guard_path[g_start_t]
        ex, ey = guard_path[g_end_t]
        ax.plot(sx, sy, marker="s", markersize=5, color="black")  # guard start
        ax.plot(ex, ey, marker="s", markersize=5, color="gray")   # guard end

    # --- Player path segments ---
    for i in range(len(player_positions) - 1):
        (px0, py0) = player_positions[i]
        (px1, py1) = player_positions[i + 1]
        t = timesteps[i]
        ax.plot(
            [px0, px1],
            [py0, py1],
            color=cmap(norm(t)),
            linewidth=2.5,
            alpha=1.0,
        )

    # Player start/end markers (no labels here; legend will be custom)
    sx, sy = player_positions[0]
    ex, ey = player_positions[-1]
    ax.plot(sx, sy, "bo", markersize=6)
    ax.plot(ex, ey, "go", markersize=6)

    # --- Mark "spotted" positions (player visible at timestep t) ---
    spotted_x, spotted_y = [], []
    for (t, pos) in zip(timesteps, player_positions):
        if map_obj.is_visible(pos, t):
            spotted_x.append(pos[0])
            spotted_y.append(pos[1])

    if spotted_x:
        ax.scatter(
            spotted_x,
            spotted_y,
            marker="x",
            s=60,
            color="red",
            linewidths=1.5,
        )

    # --- Colourbar for timesteps ---
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Timestep")

    # --- Custom legend so we only get one entry per symbol ---
    legend_elements = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='blue', markeredgecolor='blue',
               markersize=7, label='Player start'),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='green', markeredgecolor='green',
               markersize=7, label='Player end'),
        Line2D([0], [0], marker='x', linestyle='None',
               color='red', markersize=8, label='Spotted'),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor='black', markeredgecolor='black',
               markersize=7, label='Guard start'),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=7, label='Guard end'),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),   # above title
        ncol=3,
        framealpha=0.9,
        fontsize=9,
    )

    ax.margins(0.05)
    plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        print(f"Saved colour-coded map to {out_path}")



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
            ax.plot(kx, ky, "ro", markersize=4, alpha=0.3)
            ax.text(kx+0.05, ky+0.05, f"{k.get_depth()}", fontsize=6)

        # draw path so far
        path_pts.extend(list(edge._path.coords))
        xs, ys = zip(*path_pts)
        ax.plot(xs, ys, "g--", lw=2, alpha=0.3)

        ax.set_title(f"{label}: t={node._depth}, Q/N={node._total_value/node._visits:.2f}, N={node._visits}")

        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

    make_gifs(save_dir)

    return in_shadow, len(path)


def output_results(tree: MonteCarloGraph, run_time: str, save_dir="plots/best_path_gif", duration=0.25, save=True):
    os.makedirs(save_dir, exist_ok=True)
    p_visits   = os.path.join(save_dir, "best_path_max_visits")

    # extract all 3 paths
    path_visits = tree.get_path_max_visits()

    # plot each
    shadow_ct1, total1 = plot_path(tree, path_visits, p_visits, "Max-visits")

    static_path_png = os.path.join(save_dir, "best_path_max_visits_static.png")
    plot_static_colored_paths_with_spotted(tree, path_visits, static_path_png, show=False)

    # write stats
    stats = tree.get_stats()
    stats += f"Max-visits path gives {shadow_ct1} of {total1} ({shadow_ct1/total1*100:.2f}%) steps in shadow\n"
    stats += f"Took {run_time} to find the best path\n"

    with open(os.path.join(save_dir, "stats.txt"), 'w') as f:
        f.write(stats)
    
    if save:
        csv_path = os.path.join(save_dir, "..", "visibility_experiments_set_starts.csv")
        file_exists = os.path.isfile(csv_path)

        header = [
            "Level",
            "Runtime (s)",
            "C (exploration constant)",

            "k (progressive widening constant)",
            "alpha (progressive widening exponent)", 
            
            "delta (visible area distance)",
            "theta (shadow distance)",
            "beta (guard distance)", 
            "gamma (kernel distance)",
            "tau (kernel decay rate)",
            "top_k (kernels scored)",
            "eta (depth bonus)",
            
            "lambda (rollout weight)",
            "phi (rollout depth)",
            "rollout_ratio",

            "visible node penalty",

            "brute_force_moves",
            "path_cache_hits","num_path_queries","path_cache_hit_ratio",
            "tt_hits","tt_queries","tt_hit_ratio",
            "player_start_pos",

            "visits_timesteps_shadow","visits_total_timesteps","visits_shadow_pct"
        ]


        # compute values
        row = {
            "Level": tree._map._level,
            "Runtime (s)": float(run_time),

            "C (exploration constant)": tree._c,
            "k (progressive widening constant)": tree._k,
            "alpha (progressive widening exponent)": tree._alpha,       
            "delta (visible area distance)": tree._delta,
            "theta (shadow distance)": tree._theta,
            "beta (guard distance)": tree._beta,
            "gamma (kernel distance)": tree._gamma,
            "tau (kernel decay rate)": tree._tau,
            "top_k (kernels scored)": tree._top_k,
            "eta (depth bonus)": tree._eta,
            "lambda (rollout weight)": tree._lambda,
            "phi (rollout depth)": tree._phi,
            "rollout_ratio": tree._rollout_ratio,

            "visible node penalty": tree._visible_node_penalty,
            "brute_force_moves": tree._brute_force,

            "path_cache_hits": tree._map._path_cache_hits,
            "num_path_queries": tree._map._path_cache_queries,
            "path_cache_hit_ratio": (
                round(tree._map._path_cache_hits / tree._map._path_cache_queries * 100, 2)
                if tree._map._path_cache_queries > 0 else 0
            ),
            "tt_hits": tree._tt_hits,
            "tt_queries": tree._tt_queries,
            "tt_hit_ratio": (
                round(tree._tt_hits / tree._tt_queries * 100, 2)
                if tree._tt_queries > 0 else 0
            ),

            "player_start_pos": tree._map.get_player_start_pos(),

            "visits_timesteps_shadow": shadow_ct1,
            "visits_total_timesteps": total1,
            "visits_shadow_pct": round(shadow_ct1 / total1 * 100, 2) if total1 > 0 else 0,
        }


        # write to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)

            # write header only if new file
            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

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
        shadow = map.get_shadow(t)
        add_polygon(ax, shadow, fc="blue", ec="black", alpha=0.25)

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


import matplotlib.pyplot as plt
from shapely.ops import unary_union

def plot_shadow_variants(map: Map, timestep: int, save_path=None, figsize=(10, 5)):
    """
    Plot the two shadow representations for a single timestep:
      (1) shadow without obstacles
      (2) shadow with obstacles merged

    Produces a side-by-side comparison suitable for a paper figure.
    """

    boundary = map._shapely_boundary
    obstacles = unary_union(map._shapely_obstacles)

    shadow = map._shadows[timestep]
    shadow_w_obs = map._shadows_w_obs[timestep]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    titles = [
        "Shadow region (obstacles excluded)",
        "Shadow region (obstacles included)"
    ]

    for ax, shadow_poly, title in zip(axes, [shadow, shadow_w_obs], titles):
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Map geometry
        add_polygon(ax, boundary, fc="white", ec="black", alpha=1.0, zorder=1)
        add_polygon(ax, obstacles, fc="dimgray", ec="black", alpha=1.0, zorder=2)

        # Shadow
        add_polygon(ax, shadow_poly, fc="blue", alpha=0.35, zorder=3)

        # Guards
        for i, guard in enumerate(map._guards):
            gx, gy = guard.get_path()[timestep]
            ax.plot(gx, gy, "r^", markersize=6)

        ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved shadow comparison → {save_path}")
    else:
        plt.show()

import os
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def plot_shadow_kernels_side_by_side(
    map: Map,
    timestep: int,
    save_path: str | None = None,
    figsize=(12, 5)
):
    """
    Plot two subfigures for a single timestep:
      Left:  shadow (no obstacles) + kernels from that shadow
      Right: shadow with obstacles merged + kernels from that shadow

    Kernel points are annotated with their recursive depth.
    """

    boundary = map._shapely_boundary
    obstacles = unary_union(map._shapely_obstacles)

    shadow = map._shadows[timestep]
    shadow_w_obs = map._shadows_w_obs[timestep]

    # Ensure kernels are computed
    kernels_no_obs = map._kernels[timestep] if map._kernels else map.get_kernels(timestep)
    kernels_w_obs = map._kernels_w_obs[timestep] if map._kernels_w_obs else map.get_kernels(timestep)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    panels = [
        (
            axes[0],
            shadow,
            kernels_no_obs,
            "Shadow (obstacles excluded)",
            "green"
        ),
        (
            axes[1],
            shadow_w_obs,
            kernels_w_obs,
            "Shadow (obstacles included)",
            "orange"
        ),
    ]

    for ax, shadow_poly, kernels, title, kernel_color in panels:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # -------------------------------------------------------
        # Map geometry
        # -------------------------------------------------------
        add_polygon(ax, boundary, fc="white", ec="black", alpha=1.0, zorder=1)
        add_polygon(ax, obstacles, fc="dimgray", ec="black", alpha=1.0, zorder=2)

        # -------------------------------------------------------
        # Shadow
        # -------------------------------------------------------
        add_polygon(ax, shadow_poly, fc="blue", alpha=0.35, zorder=3)

        # -------------------------------------------------------
        # Kernels + depth annotations
        # -------------------------------------------------------
        for k in kernels:
            x, y = k.get_coords()
            depth = k.get_depth()

            ax.plot(
                x, y,
                marker="o",
                color=kernel_color,
                markersize=6,
                zorder=5
            )

            ax.annotate(
                str(depth),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                color="black",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.75
                )
            )


        # -------------------------------------------------------
        # Guard positions
        # -------------------------------------------------------
        for guard in map._guards:
            gx, gy = guard.get_path()[timestep]
            ax.plot(gx, gy, "r^", markersize=7, zorder=7)

        ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved shadow + kernel comparison → {save_path}")
    else:
        plt.show()


def plot_static_guard_paths_by_timestep(
    map_obj,
    out_path,
    show=False,
    figsize=(7, 7),
    cmap=plt.cm.plasma
):
    """
    Plot a static, colour-coded visualization of guard paths.
    Each guard's trajectory is coloured by timestep to show motion over time.

    No player paths or visibility markers are shown.
    """

    # -------------------------------------------------------
    # Setup figure
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # -------------------------------------------------------
    # Draw map geometry
    # -------------------------------------------------------
    add_polygon(ax, map_obj._shapely_boundary, fc="white", ec="black", alpha=1.0)

    if map_obj._shapely_obstacles:
        add_polygon(
            ax,
            unary_union(map_obj._shapely_obstacles),
            fc="dimgray",
            ec="black",
            alpha=1.0,
        )

    # -------------------------------------------------------
    # Determine timestep range
    # -------------------------------------------------------
    T = map_obj.get_num_timesteps()
    t_min, t_max = 0, max(T - 1, 1)
    norm = plt.Normalize(vmin=t_min, vmax=t_max)

    # -------------------------------------------------------
    # Plot guard paths
    # -------------------------------------------------------
    for guard_id, guard in enumerate(map_obj._guards):
        path = guard.get_path()
        T_guard = len(path)

        # Draw coloured segments
        for t in range(T_guard - 1):
            (x0, y0) = path[t]
            (x1, y1) = path[t + 1]

            ax.plot(
                [x0, x1],
                [y0, y1],
                color=cmap(norm(t)),
                linewidth=2.0,
                alpha=0.9,
            )

        # Mark start and end
        sx, sy = path[0]
        ex, ey = path[-1]

        ax.plot(sx, sy, marker="s", markersize=6, color="black")
        ax.plot(ex, ey, marker="s", markersize=6, color="gray")

    # -------------------------------------------------------
    # Colourbar
    # -------------------------------------------------------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Timestep", fontsize=10)

    # -------------------------------------------------------
    # Legend (minimal, clean)
    # -------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor='black', markeredgecolor='black',
               markersize=7, label='Guard start'),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=7, label='Guard end'),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        framealpha=0.9,
        fontsize=9,
    )

    ax.margins(0.05)
    plt.tight_layout()

    # -------------------------------------------------------
    # Save / show
    # -------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        print(f"Saved guard-path visualization to {out_path}")
