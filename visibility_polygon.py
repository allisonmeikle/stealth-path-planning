import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# ----------------------------
# 1. Map and obstacles
# ----------------------------
map_poly = Polygon([(0,0), (6,0), (6,6), (0,6)])

# Four 1x1 obstacles, one tile in from each corner
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
guard = Point(0.01, 0.01)  # slightly inside top-left corner
facing_angle = 0        # 0 radians = right
fov = math.pi           # 180 degrees
num_rays = 200
max_distance = 10

# ----------------------------
# 3. Build segments for ray-casting
# ----------------------------
def polygon_to_segments(poly):
    coords = list(poly.exterior.coords)
    return [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]

segments = polygon_to_segments(map_poly)
for obs in obstacles:
    segments += polygon_to_segments(obs)

# ----------------------------
# 4. Ray casting function
# ----------------------------
def cast_ray(guard, angle, segments):
    dx = math.cos(angle)*max_distance
    dy = math.sin(angle)*max_distance
    ray = LineString([guard.coords[0], (guard.x+dx, guard.y+dy)])
    min_dist = float('inf')
    closest_point = ray.coords[1]
    for seg in segments:
        inter = ray.intersection(seg)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            d = guard.distance(inter)
            if d < min_dist:
                min_dist = d
                closest_point = (inter.x, inter.y)
        elif inter.geom_type == "MultiPoint":
            for p in inter.geoms:
                d = guard.distance(p)
                if d < min_dist:
                    min_dist = d
                    closest_point = (p.x, p.y)
    return closest_point

# ----------------------------
# 5. Cast rays for each degree in FOV
# ----------------------------
angles = [facing_angle - fov/2 + i*(fov/num_rays) for i in range(num_rays+1)]
ray_endpoints = [cast_ray(guard, angle, segments) for angle in angles]

# ----------------------------
# 6. Build visibility polygon
# ----------------------------
vis_poly = Polygon(ray_endpoints)

# ----------------------------
# 7. Compute shadow/safe area (still store, but don't plot)
# ----------------------------
shadow_area = map_free.difference(vis_poly)

if shadow_area.geom_type == 'Polygon':
    print("Shadow area is a single polygon.")
    shadow_polys = [shadow_area]
elif shadow_area.geom_type == 'MultiPolygon':
    print("Shadow area is multiple polygon.")
    shadow_polys = list(shadow_area.geoms)
else:
    shadow_polys = []

# ----------------------------
# 8. Plot everything
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
xv, yv = vis_poly.exterior.xy
plt.fill(xv, yv, color='red', alpha=0.3)

# Grid
for i in range(7):
    plt.plot([i,i],[0,6], color='black', linestyle='--', linewidth=0.5)
    plt.plot([0,6],[i,i], color='black', linestyle='--', linewidth=0.5)

plt.xlim(0,6)
plt.ylim(0,6)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("6x6 Map with Guard Visibility (Red)")
plt.show()
