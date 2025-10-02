import os
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
from descartes import PolygonPatch

plt.style.use("classic")

def add_polygon(ax, poly, **kwargs):
    if poly.is_empty:
        return
    if poly.geom_type == "Polygon":
        if poly.is_valid and len(poly.exterior.coords) >= 3:
            try:
                patch = PolygonPatch(poly, **kwargs)
                ax.add_patch(patch)
            except Exception:
                # fallback if PolygonPatch fails
                x, y = poly.exterior.xy
                ax.fill(x, y, **kwargs)
    elif poly.geom_type == "MultiPolygon":
        for subpoly in poly.geoms:
            add_polygon(ax, subpoly, **kwargs)

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
    os.makedirs('maps', exist_ok=True)

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
        out_path = os.path.join('maps', file_name)
        plt.savefig(out_path, dpi=150)
        plt.close()
    else: 
        plt.show()