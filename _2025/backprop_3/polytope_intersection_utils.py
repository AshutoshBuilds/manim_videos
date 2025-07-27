import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import split, unary_union
import shapely.affinity


def intersect_polytopes(polygons_1: List[np.ndarray], polygons_2: List[np.ndarray]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Intersect two polytopes represented as lists of polygons.
    
    Args:
        polygons_1: List of Nx3 numpy arrays representing first polytope
        polygons_2: List of Nx3 numpy arrays representing second polytope
        
    Returns:
        - intersection_lines: List of (start, end) 3D points for intersection lines
        - new_2d_tiling: List of 2D polygons (Nx2 arrays) after splitting
        - top_polytope: List of 3D polygons (Nx3 arrays) representing the top surface
        - indicator: 1D array (0 if from polygons_1, 1 if from polygons_2)
    """
    assert len(polygons_1) == len(polygons_2), "Polygon lists must have same length"
    
    intersection_lines = []
    all_split_polygons = []  # Will store (2d_poly, orig_idx, intersection_line) tuples
    
    # Process each polygon pair
    for idx, (poly1, poly2) in enumerate(zip(polygons_1, polygons_2)):
        # Extract 2D coordinates (x, y) - should be same for both polygons
        xy_coords = poly1[:, :2]
        
        # Get z values for both polygons
        z1 = poly1[:, 2]
        z2 = poly2[:, 2]
        
        # Check if one polygon is completely above the other
        if np.all(z1 >= z2):
            # polygon_1 is completely on top
            all_split_polygons.append((xy_coords, idx, None))
        elif np.all(z2 >= z1):
            # polygon_2 is completely on top
            all_split_polygons.append((xy_coords, idx, None))
        else:
            # Polygons intersect - find intersection line
            intersection_points = find_polygon_intersection(poly1, poly2)
            
            if len(intersection_points) == 2:
                # Add to intersection lines
                intersection_lines.append((intersection_points[0], intersection_points[1]))
                
                # Create Shapely polygon
                shapely_poly = Polygon(xy_coords)
                
                # Create intersection line in 2D
                line_2d = LineString([intersection_points[0][:2], intersection_points[1][:2]])
                
                # Split the polygon using Shapely
                split_result = split(shapely_poly, line_2d)
                
                if isinstance(split_result, (MultiPolygon, list)):
                    # Successfully split
                    for geom in split_result.geoms if hasattr(split_result, 'geoms') else split_result:
                        if isinstance(geom, Polygon) and geom.is_valid:
                            # Extract coordinates
                            coords = np.array(geom.exterior.coords[:-1])  # Remove duplicate last point
                            all_split_polygons.append((coords, idx, line_2d))
                else:
                    # Split failed, keep original
                    all_split_polygons.append((xy_coords, idx, None))
            else:
                # No valid intersection, keep original
                all_split_polygons.append((xy_coords, idx, None))
    
    # Now build the final tiling and 3D polytope
    new_2d_tiling = []
    top_polytope = []
    indicator = []
    
    for poly_2d, orig_idx, split_line in all_split_polygons:
        poly1 = polygons_1[orig_idx]
        poly2 = polygons_2[orig_idx]
        
        # Add to 2D tiling
        new_2d_tiling.append(poly_2d)
        
        # Create 3D polygon and determine which surface is on top
        poly_3d, is_poly2_on_top = create_3d_top_polygon(poly_2d, poly1, poly2)
        
        top_polytope.append(poly_3d)
        indicator.append(1 if is_poly2_on_top else 0)
    
    return intersection_lines, new_2d_tiling, top_polytope, np.array(indicator)


def find_polygon_intersection(poly1: np.ndarray, poly2: np.ndarray) -> List[np.ndarray]:
    """Find intersection points between two polygons with same x,y but different z."""
    intersection_points = []
    n = len(poly1)
    
    # Check each edge
    for i in range(n):
        j = (i + 1) % n
        
        # Edge endpoints
        p1_i, p1_j = poly1[i], poly1[j]
        p2_i, p2_j = poly2[i], poly2[j]
        
        # Z values at endpoints
        z1_i, z1_j = p1_i[2], p1_j[2]
        z2_i, z2_j = p2_i[2], p2_j[2]
        
        # Check if surfaces cross on this edge
        diff_i = z1_i - z2_i
        diff_j = z1_j - z2_j
        
        if diff_i * diff_j < 0:  # Different signs = crossing
            # Linear interpolation to find crossing point
            t = diff_i / (diff_i - diff_j)
            intersection_3d = p1_i + t * (p1_j - p1_i)
            intersection_points.append(intersection_3d)
    
    return intersection_points


def create_3d_top_polygon(poly_2d: np.ndarray, orig_poly1: np.ndarray, orig_poly2: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Create a 3D polygon by choosing the higher surface at each point.
    Returns the 3D polygon and a boolean indicating if poly2 is on top.
    """
    n = len(poly_2d)
    poly_3d = np.zeros((n, 3))
    
    # Copy x,y coordinates
    poly_3d[:, :2] = poly_2d
    
    # Create Shapely polygons for point-in-polygon tests
    orig_2d_poly = Polygon(orig_poly1[:, :2])
    
    # Determine which surface is on top at the center
    center = np.mean(poly_2d, axis=0)
    center_point = Point(center)
    
    # Check if center is inside original polygon (it should be)
    if not orig_2d_poly.contains(center_point):
        # Find a point that is definitely inside
        center = poly_2d[0]  # Use first vertex as fallback
    
    # Interpolate z values at center
    z1_center = interpolate_z_at_point(center, orig_poly1[:, :2], orig_poly1[:, 2])
    z2_center = interpolate_z_at_point(center, orig_poly2[:, :2], orig_poly2[:, 2])
    
    # Use the surface that's higher at the center
    use_poly2 = z2_center > z1_center
    
    # Interpolate z values for all vertices
    for i in range(n):
        point = poly_2d[i]
        
        if use_poly2:
            poly_3d[i, 2] = interpolate_z_at_point(point, orig_poly2[:, :2], orig_poly2[:, 2])
        else:
            poly_3d[i, 2] = interpolate_z_at_point(point, orig_poly1[:, :2], orig_poly1[:, 2])
    
    return poly_3d, use_poly2


def interpolate_z_at_point(point_2d: np.ndarray, polygon_2d: np.ndarray, z_values: np.ndarray) -> float:
    """
    Interpolate z-value at a 2D point using barycentric interpolation.
    """
    # Check if point is exactly at a vertex
    for i, vertex in enumerate(polygon_2d):
        if np.linalg.norm(point_2d - vertex) < 1e-10:
            return z_values[i]
    
    # Try barycentric interpolation by triangulating from centroid
    centroid = np.mean(polygon_2d, axis=0)
    n = len(polygon_2d)
    
    for i in range(n):
        j = (i + 1) % n
        
        # Form triangle with centroid and edge
        tri_points = np.array([centroid, polygon_2d[i], polygon_2d[j]])
        tri_z = np.array([np.mean(z_values), z_values[i], z_values[j]])
        
        # Check if point is in this triangle
        bary = barycentric_coords(point_2d, tri_points)
        
        if np.all(bary >= -1e-10) and np.all(bary <= 1 + 1e-10) and np.abs(np.sum(bary) - 1) < 1e-10:
            # Point is in triangle
            return np.dot(bary, tri_z)
    
    # Fallback: inverse distance weighting
    distances = np.linalg.norm(polygon_2d - point_2d, axis=1)
    weights = 1.0 / (distances + 1e-10)
    weights /= np.sum(weights)
    
    return np.sum(weights * z_values)


def barycentric_coords(p: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Calculate barycentric coordinates of point p with respect to triangle."""
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = p - triangle[0]
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, w, v])


# Alternative implementation using shapely's prepared geometry for better performance
def intersect_polytopes_optimized(polygons_1: List[np.ndarray], polygons_2: List[np.ndarray]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Optimized version that processes all polygons together to ensure consistent splits.
    """
    assert len(polygons_1) == len(polygons_2), "Polygon lists must have same length"
    
    # Collect all intersection lines first
    all_intersection_lines = []
    intersecting_indices = []
    
    for idx, (poly1, poly2) in enumerate(zip(polygons_1, polygons_2)):
        z1 = poly1[:, 2]
        z2 = poly2[:, 2]
        
        if not (np.all(z1 >= z2) or np.all(z2 >= z1)):
            intersection_points = find_polygon_intersection(poly1, poly2)
            if len(intersection_points) == 2:
                all_intersection_lines.append((intersection_points[0], intersection_points[1]))
                intersecting_indices.append(idx)
    
    # Create a unified polygon collection and split by all intersection lines
    all_polygons = []
    polygon_to_original_idx = []
    
    for idx, poly1 in enumerate(polygons_1):
        shapely_poly = Polygon(poly1[:, :2])
        all_polygons.append(shapely_poly)
        polygon_to_original_idx.append(idx)
    
    # Split all polygons by all intersection lines
    split_polygons = all_polygons.copy()
    
    for line_3d in all_intersection_lines:
        line_2d = LineString([line_3d[0][:2], line_3d[1][:2]])
        new_split_polygons = []
        new_indices = []
        
        for poly, orig_idx in zip(split_polygons, polygon_to_original_idx):
            try:
                result = split(poly, line_2d)
                if isinstance(result, (MultiPolygon, list)):
                    for geom in result.geoms if hasattr(result, 'geoms') else result:
                        if isinstance(geom, Polygon) and geom.is_valid:
                            new_split_polygons.append(geom)
                            new_indices.append(orig_idx)
                else:
                    new_split_polygons.append(poly)
                    new_indices.append(orig_idx)
            except:
                new_split_polygons.append(poly)
                new_indices.append(orig_idx)
        
        split_polygons = new_split_polygons
        polygon_to_original_idx = new_indices
    
    # Build final results
    new_2d_tiling = []
    top_polytope = []
    indicator = []
    
    for poly, orig_idx in zip(split_polygons, polygon_to_original_idx):
        # Convert back to numpy array
        coords = np.array(poly.exterior.coords[:-1])
        new_2d_tiling.append(coords)
        
        # Create 3D polygon
        poly_3d, is_poly2_on_top = create_3d_top_polygon(
            coords, polygons_1[orig_idx], polygons_2[orig_idx]
        )
        
        top_polytope.append(poly_3d)
        indicator.append(1 if is_poly2_on_top else 0)
    
    return all_intersection_lines, new_2d_tiling, top_polytope, np.array(indicator)


# Example usage:
if __name__ == "__main__":
    # Create example polygons (squares in x-y plane with different z values)
    square_xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    # First polytope - tilted upward in x direction
    poly1 = np.column_stack([square_xy, np.array([0, 0.5, 0.5, 0])])
    
    # Second polytope - tilted upward in y direction  
    poly2 = np.column_stack([square_xy, np.array([0, 0, 0.5, 0.5])])
    
    polygons_1 = [poly1]
    polygons_2 = [poly2]
    
    # Run intersection
    intersection_lines, new_2d_tiling, top_polytope, indicator = intersect_polytopes(
        polygons_1, polygons_2
    )
    
    print("Intersection lines:", len(intersection_lines))
    print("New 2D tiling polygons:", len(new_2d_tiling))
    print("Top polytope polygons:", len(top_polytope))
    print("Indicator array:", indicator)
    
    # Also test the optimized version
    print("\nOptimized version:")
    intersection_lines, new_2d_tiling, top_polytope, indicator = intersect_polytopes_optimized(
        polygons_1, polygons_2
    )
    
    print("Intersection lines:", len(intersection_lines))
    print("New 2D tiling polygons:", len(new_2d_tiling))
    print("Top polytope polygons:", len(top_polytope))
    print("Indicator array:", indicator)