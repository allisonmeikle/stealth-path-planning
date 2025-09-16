from typing import List
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, LineString

class Shadow:
    def __init__(self, shape : BaseGeometry):
        self.shape = shape
        