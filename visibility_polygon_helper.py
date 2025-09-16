import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import visilibity as vis

# ----------------------------
# Helper functions
# ----------------------------
def shapely_to_visilibity_polygon(poly: Polygon, epsilon=1e-5):
    if not poly.is_valid:
        raise ValueError("Invalid shapely polygon")

    # Force outer ring CCW
    poly = orient(poly, sign=1.0)

    # Outer boundary
    exterior_coords = list(poly.exterior.coords)
    outer = vis.Polygon([vis.Point(x, y) for x, y in exterior_coords])

    # Force holes CW
    holes = []
    for hole in poly.interiors:
        hole_coords = list(hole.coords)
        # orient() can’t be applied directly to LinearRing, so flip manually
        if Polygon(hole_coords).exterior.is_ccw:
            hole_coords = list(reversed(hole_coords))
        hpoly = vis.Polygon([vis.Point(x, y) for x, y in hole_coords])
        holes.append(hpoly)

    return outer, holes

def compute_visibility_polygon(map_poly, obstacles, guard_pos, epsilon=1e-5):
    guard = vis.Point(*guard_pos)

    # Outer boundary
    map_poly = orient(map_poly, sign=1.0)
    outer_coords = list(map_poly.exterior.coords)[:-1]
    outer = vis.Polygon([vis.Point(x, y) for x, y in outer_coords])

    # Obstacles
    holes = []
    for obs in obstacles:
        hole_coords = list(obs.exterior.coords)[:-1]
        if Polygon(hole_coords).exterior.is_ccw:
            hole_coords = list(reversed(hole_coords))
        hpoly = vis.Polygon([vis.Point(x, y) for x, y in hole_coords])
        holes.append(hpoly)

    env = vis.Environment([outer] + holes)

    if not env.is_valid():
        raise ValueError("Invalid VisiLibity environment")
    if hasattr(env, "point_in_environment") and not env.point_in_environment(guard, epsilon):
        raise ValueError("Guard point not inside environment")

    V = vis.Visibility_Polygon(guard, env, epsilon)

    # FIX: iterate instead of len(V)
    coords = [(p.x(), p.y()) for p in V]
    return Polygon(coords)

# ----------------------------
# 1. Map and obstacles
# ----------------------------
map_poly = Polygon([(0,0), (6,0), (6,6), (0,6)])

obstacles = [
    Polygon([(1,1), (2,1), (2,2), (1,2)]),
    Polygon([(4,1), (5,1), (5,2), (4,2)]),
    Polygon([(1,4), (2,4), (2,5), (1,5)]),
    Polygon([(4,4), (5,4), (5,5), (4,5)])
]

# Free map area = map minus obstacles
map_free = map_poly.difference(unary_union(obstacles))

# ----------------------------
# 2. Guard setup
# ----------------------------
guard = Point(0.5, 0.5)  # must be inside free area!

# ----------------------------
# 3. Build environment polygon (map with holes)
# ----------------------------
map_with_holes = Polygon(shell=map_poly.exterior.coords,
                         holes=[obs.exterior.coords for obs in obstacles])

# ----------------------------
# 4. Compute visibility polygon with VisiLibity
# ----------------------------
vis_poly = compute_visibility_polygon(map_poly, obstacles, (guard.x, guard.y))

# ----------------------------
# 5. Compute shadow area (map minus visibility polygon)
# ----------------------------
shadow_area = map_free.difference(vis_poly)

if shadow_area.geom_type == 'Polygon':
    shadow_polys = [shadow_area]
elif shadow_area.geom_type == 'MultiPolygon':
    shadow_polys = list(shadow_area.geoms)
else:
    shadow_polys = []

# ----------------------------
# 6. Plot everything
# ----------------------------
plt.figure(figsize=(6,6))

# Map background
x, y = map_poly.exterior.xy
plt.fill(x, y, color='lightgrey', alpha=0.5)

# Obstacles
for obs in obstacles:
    xo, yo = obs.exterior.xy
    plt.fill(xo, yo, color='darkgrey', alpha=0.8)

# Guard
plt.plot(guard.x, guard.y, 'ro', markersize=8)

# Visibility polygon in red
if not vis_poly.is_empty:
    xv, yv = vis_poly.exterior.xy
    plt.fill(xv, yv, color='red', alpha=0.3)

# Grid
for i in range(7):
    plt.plot([i,i],[0,6], color='black', linestyle='--', linewidth=0.5)
    plt.plot([0,6],[i,i], color='black', linestyle='--', linewidth=0.5)

plt.xlim(0,6)
plt.ylim(0,6)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("6x6 Map with Guard Visibility (Red, via VisiLibity)")
plt.show()
