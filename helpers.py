import math

def same_position(p1: tuple[float, float], p2: tuple[float, float], tol: float = 1e-9) -> bool:
    return math.isclose(p1[0], p2[0], abs_tol=tol) and math.isclose(p1[1], p2[1], abs_tol=tol)
