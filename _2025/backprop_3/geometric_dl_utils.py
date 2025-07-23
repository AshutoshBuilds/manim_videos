from manimlib import *
from functools import partial
from itertools import combinations
import math
import torch.nn as nn
import torch

class BaarleNet(nn.Module):
    def __init__(self, hidden_layers=[64]):
        super(BaarleNet, self).__init__()
        layers = [nn.Linear(2, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.layers=layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def surface_func_from_model(u, v, model, layer_idx, neuron_idx, viz_scale=0.5):
    """
    Create a surface function for visualizing activations at any layer.
    
    Args:
        u, v: Input coordinates 
        model: BaarleNet model
        layer_idx: Direct index into model.model (e.g., 0, 1, 2, 3...)
        neuron_idx: Which neuron in that layer
        viz_scale: Scaling factor for visualization
    """
    input_tensor = torch.tensor([[u, v]], dtype=torch.float32)
    
    with torch.no_grad():
        x = input_tensor
        # Forward through layers up to and including target layer
        for i in range(layer_idx + 1):
            x = model.model[i](x)
        
        activation = x[0, neuron_idx].item()
        z = activation * viz_scale
        return np.array([u, v, z])


def get_polygon_corners_layer_1(model):
    """
    Extract ReLU boundary lines from first layer and create polygons for each neuron.
    
    Args:
        model: PyTorch model with Sequential layers
        
    Returns:
        list: List of polygon corner points for each neuron in first layer
              Each element contains two polygons (positive and negative regions)
    """
    # Get first layer weights and biases
    first_layer = model.model[0]  # Linear layer
    weights = first_layer.weight.detach().numpy()  # Shape: (out_features, in_features)
    biases = first_layer.bias.detach().numpy()     # Shape: (out_features,)
    
    # Define the boundary of our region [-1, 1, -1, 1] -> [x_min, x_max, y_min, y_max]
    boundary = [-1, 1, -1, 1]
    x_min, x_max, y_min, y_max = boundary
    
    # Corner points of the original square
    square_corners = [
        [x_min, y_min],  # bottom-left
        [x_max, y_min],  # bottom-right
        [x_max, y_max],  # top-right
        [x_min, y_max]   # top-left
    ]
    
    polygons_per_neuron = []
    
    # Process each neuron in the first layer
    for neuron_idx in range(weights.shape[0]):
        w1, w2 = weights[neuron_idx]  # weights for x and y
        b = biases[neuron_idx]        # bias
        
        # ReLU boundary line equation: w1*x + w2*y + b = 0
        # Rearrange to: y = -(w1*x + b) / w2 (if w2 != 0)
        
        # Find intersection points with boundary
        intersections = []
        
        # Check intersection with each edge of the square
        edges = [
            ([x_min, y_min], [x_max, y_min]),  # bottom edge
            ([x_max, y_min], [x_max, y_max]),  # right edge
            ([x_max, y_max], [x_min, y_max]),  # top edge
            ([x_min, y_max], [x_min, y_min])   # left edge
        ]
        
        for edge_start, edge_end in edges:
            # Line-line intersection
            if edge_start[0] == edge_end[0]:  # vertical edge
                x = edge_start[0]
                if w2 != 0:
                    y = -(w1 * x + b) / w2
                    if y_min <= y <= y_max:
                        intersections.append([x, y])
            else:  # horizontal edge
                y = edge_start[1]
                if w1 != 0:
                    x = -(w2 * y + b) / w1
                    if x_min <= x <= x_max:
                        intersections.append([x, y])
        
        # Remove duplicate intersections (within tolerance)
        unique_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if abs(point[0] - existing[0]) < 1e-6 and abs(point[1] - existing[1]) < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(point)
        
        if len(unique_intersections) >= 2:
            # Sort intersections to create proper polygon ordering
            # Use the first two intersections to define the line
            p1, p2 = unique_intersections[0], unique_intersections[1]
            
            # Determine which side of the line each corner is on
            def side_of_line(point, line_p1, line_p2):
                # Cross product to determine side
                return (line_p2[0] - line_p1[0]) * (point[1] - line_p1[1]) - \
                       (line_p2[1] - line_p1[1]) * (point[0] - line_p1[0])
            
            positive_corners = []
            negative_corners = []
            
            # Classify each corner of the square
            for corner in square_corners:
                # Check which side of ReLU line this corner is on
                activation = w1 * corner[0] + w2 * corner[1] + b
                if activation >= 0:
                    positive_corners.append(corner)
                else:
                    negative_corners.append(corner)
            
            # Add intersection points to both polygons
            positive_polygon = positive_corners + unique_intersections
            negative_polygon = negative_corners + unique_intersections
            
            # Sort points to form proper polygons (counter-clockwise)
            def sort_polygon_points(points):
                if len(points) < 3:
                    return points
                
                # Find centroid
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                
                # Sort by angle from centroid
                def angle_from_center(point):
                    return np.arctan2(point[1] - cy, point[0] - cx)
                
                return sorted(points, key=angle_from_center)
            
            positive_polygon = sort_polygon_points(positive_polygon)
            negative_polygon = sort_polygon_points(negative_polygon)
            
            polygons_per_neuron.append({
                'positive_region': positive_polygon,
                'negative_region': negative_polygon,
                'relu_line': unique_intersections,
                'line_equation': f"{w1:.3f}*x + {w2:.3f}*y + {b:.3f} = 0"
            })
        else:
            # If line doesn't intersect the square properly, return the whole square for positive
            # and empty for negative (or vice versa based on bias)
            if b >= 0:  # If bias is positive, most of square might be positive
                polygons_per_neuron.append({
                    'positive_region': square_corners,
                    'negative_region': [],
                    'relu_line': [],
                    'line_equation': f"{w1:.3f}*x + {w2:.3f}*y + {b:.3f} = 0"
                })
            else:
                polygons_per_neuron.append({
                    'positive_region': [],
                    'negative_region': square_corners,
                    'relu_line': [],
                    'line_equation': f"{w1:.3f}*x + {w2:.3f}*y + {b:.3f} = 0"
                })
    
    return polygons_per_neuron


def carve_plane_with_relu_joints(joint_points_list, extent=1):
    """
    Compute the corner points of polygons formed by multiple ReLU joint lines.
    Each line divides the plane into two half-planes, creating distinct regions.
    
    Args:
        joint_points_list: List of joint point pairs, one for each neuron
        extent: The boundary of the square domain
    
    Returns:
        List of polygons, where each polygon is a list of corner points
    """
    # Filter out empty joint lines
    valid_lines = []
    for joint_points in joint_points_list:
        if joint_points and len(joint_points) >= 2:
            valid_lines.append(joint_points[:2])
    
    if len(valid_lines) == 0:
        # If no lines, return the entire square
        return [[[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]]]
    
    def line_equation(p1, p2):
        """Get line equation coefficients ax + by + c = 0"""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    
    def evaluate_line(point, a, b, c):
        """Evaluate ax + by + c for a point"""
        return a * point[0] + b * point[1] + c
    
    def line_intersection(p1, p2, p3, p4):
        """Find intersection of two lines"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return [x, y]
    
    def clip_polygon_by_line(polygon, line_p1, line_p2):
        """Clip a polygon by a line using Sutherland-Hodgman algorithm"""
        if not polygon:
            return []
        
        a, b, c = line_equation(line_p1, line_p2)
        
        output_polygon = []
        n = len(polygon)
        
        for i in range(n):
            current_vertex = polygon[i]
            previous_vertex = polygon[i-1]
            
            current_side = evaluate_line(current_vertex, a, b, c)
            previous_side = evaluate_line(previous_vertex, a, b, c)
            
            # We keep points on the positive side (or on the line)
            if current_side >= -1e-10:  # Current vertex is inside
                if previous_side < -1e-10:  # Previous vertex was outside
                    # Add intersection point
                    intersection = line_intersection(
                        previous_vertex, current_vertex,
                        line_p1, line_p2
                    )
                    if intersection:
                        output_polygon.append(intersection)
                output_polygon.append(current_vertex)
            elif previous_side >= -1e-10:  # Current is outside, previous was inside
                # Add intersection point
                intersection = line_intersection(
                    previous_vertex, current_vertex,
                    line_p1, line_p2
                )
                if intersection:
                    output_polygon.append(intersection)
        
        return output_polygon
    
    # Start with the full square
    initial_square = [
        [-extent, -extent],
        [extent, -extent],
        [extent, extent],
        [-extent, extent]
    ]
    
    # For each combination of line sides, create a region
    n_lines = len(valid_lines)
    polygons = []
    
    for region_idx in range(2**n_lines):
        # Start with the full square
        current_polygon = initial_square[:]
        
        # Clip by each line
        for line_idx in range(n_lines):
            if current_polygon:
                line = valid_lines[line_idx]
                
                # Determine which side to keep based on the region code
                if region_idx & (1 << line_idx):
                    # Keep positive side - use line as is
                    current_polygon = clip_polygon_by_line(current_polygon, line[0], line[1])
                else:
                    # Keep negative side - reverse line direction
                    current_polygon = clip_polygon_by_line(current_polygon, line[1], line[0])
        
        # Add the resulting polygon if it's non-empty
        if len(current_polygon) >= 3:
            # Remove duplicate vertices
            cleaned_polygon = []
            for vertex in current_polygon:
                is_duplicate = False
                for existing in cleaned_polygon:
                    if abs(vertex[0] - existing[0]) < 1e-8 and abs(vertex[1] - existing[1]) < 1e-8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    cleaned_polygon.append(vertex)
            
            if len(cleaned_polygon) >= 3:
                polygons.append(cleaned_polygon)
    
    return polygons


def apply_relu_to_polygon(polygon):
    """
    Apply ReLU operation to a polygon's z-values (clip negative values to 0).
    
    Args:
        polygon: numpy array of shape (n_points, 3)
    
    Returns:
        numpy array with z-values clipped to be >= 0
    """
    clipped_polygon = polygon.copy()
    clipped_polygon[:, 2] = np.maximum(clipped_polygon[:, 2], 0.0)
    return clipped_polygon



import numpy as np

def split_polygons_with_relu(layer_polygons_3d):
    """
    Split 3D polygons that cross the z=0 plane (ReLU boundary) and merge zero regions.
    
    Args:
        layer_polygons_3d: List of lists of numpy arrays representing 3D polygons
                          Each sublist represents polygons for one neuron
                          Each numpy array is a polygon with shape (n_points, 3)
    
    Returns:
        Tuple of (all_polygons, merged_zero_polygons, unmerged_polygons)
        - all_polygons: List of lists of numpy arrays with split polygons (same as before)
        - merged_zero_polygons: List of lists with merged zero-height polygons
        - unmerged_polygons: List of lists with non-zero polygons that weren't merged
    """
    all_polygons = []
    merged_zero_polygons = []
    unmerged_polygons = []
    
    for neuron_idx, neuron_polygons in enumerate(layer_polygons_3d):
        neuron_all = []
        
        for polygon in neuron_polygons:
            if len(polygon) < 3:
                # Skip degenerate polygons
                neuron_all.append(polygon)
                continue
                
            # Check if polygon crosses z=0
            z_values = polygon[:, 2]
            min_z = np.min(z_values)
            max_z = np.max(z_values)
            
            if min_z >= 0 or max_z <= 0:
                # Polygon doesn't cross z=0, apply ReLU clipping and keep as is
                clipped_polygon = apply_relu_to_polygon(polygon)
                neuron_all.append(clipped_polygon)
            else:
                # Polygon crosses z=0, need to split it
                split_polygons = split_polygon_at_z_zero(polygon)
                neuron_all.extend(split_polygons)
        
        all_polygons.append(neuron_all)
        
        # Separate zero and non-zero polygons for this neuron
        zero_polygons = []
        nonzero_polygons = []
        
        for polygon in neuron_all:
            if is_zero_polygon(polygon):
                zero_polygons.append(polygon)
            else:
                nonzero_polygons.append(polygon)
        
        # Merge adjacent zero polygons
        merged_zeros = merge_adjacent_polygons(zero_polygons)
        
        merged_zero_polygons.append(merged_zeros)
        unmerged_polygons.append(nonzero_polygons)
    
    return all_polygons, merged_zero_polygons, unmerged_polygons


def split_polygon_at_z_zero(polygon):
    """
    Split a single 3D polygon at the z=0 plane.
    
    Args:
        polygon: numpy array of shape (n_points, 3) representing polygon vertices
    
    Returns:
        List of numpy arrays representing the split polygons
    """
    n_points = len(polygon)
    if n_points < 3:
        return [polygon]
    
    # Find intersection points with z=0 plane
    intersection_points = []
    intersection_indices = []  # Track where intersections occur in the original polygon
    
    for i in range(n_points):
        curr_point = polygon[i]
        next_point = polygon[(i + 1) % n_points]
        
        curr_z = curr_point[2]
        next_z = next_point[2]
        
        # Check if edge crosses z=0
        if (curr_z > 0 and next_z < 0) or (curr_z < 0 and next_z > 0):
            # Find intersection point using linear interpolation
            t = -curr_z / (next_z - curr_z)
            intersection = curr_point + t * (next_point - curr_point)
            intersection[2] = 0.0  # Ensure exactly on z=0 plane
            
            intersection_points.append(intersection)
            intersection_indices.append(i)
    
    if len(intersection_points) < 2:
        # Not enough intersections to split properly
        return [polygon]
    
    # We need exactly 2 intersection points for a clean split
    if len(intersection_points) > 2:
        # For more complex cases, keep only the first two intersections
        intersection_points = intersection_points[:2]
        intersection_indices = intersection_indices[:2]
    
    # Split polygon into two parts
    positive_polygon = []
    negative_polygon = []
    
    # Add intersection points to both polygons
    int_point_1, int_point_2 = intersection_points[0], intersection_points[1]
    
    # Traverse the original polygon and assign points to positive or negative
    for i in range(n_points):
        point = polygon[i]
        z_val = point[2]
        
        if z_val >= 0:
            positive_polygon.append(point)
        else:
            negative_polygon.append(point)
        
        # Add intersection points at the right places
        if i in intersection_indices:
            intersection_idx = intersection_indices.index(i)
            intersection_point = intersection_points[intersection_idx]
            
            # Add to both polygons
            if z_val >= 0:
                negative_polygon.append(intersection_point)
            else:
                positive_polygon.append(intersection_point)
    
    # Ensure intersection points are in both polygons
    for int_point in intersection_points:
        if len(positive_polygon) > 0 and not any(np.allclose(int_point, p, atol=1e-8) for p in positive_polygon):
            positive_polygon.append(int_point)
        if len(negative_polygon) > 0 and not any(np.allclose(int_point, p, atol=1e-8) for p in negative_polygon):
            negative_polygon.append(int_point)
    
    # Convert to numpy arrays, apply ReLU clipping, and sort points to maintain proper polygon order
    result_polygons = []
    
    if len(positive_polygon) >= 3:
        positive_polygon = np.array(positive_polygon)
        # Apply ReLU: clip z-values to be >= 0
        positive_polygon[:, 2] = np.maximum(positive_polygon[:, 2], 0.0)
        positive_polygon = sort_polygon_points_3d(positive_polygon)
        result_polygons.append(positive_polygon)
    
    if len(negative_polygon) >= 3:
        negative_polygon = np.array(negative_polygon)
        # Apply ReLU: clip z-values to be >= 0
        negative_polygon[:, 2] = np.maximum(negative_polygon[:, 2], 0.0)
        negative_polygon = sort_polygon_points_3d(negative_polygon)
        result_polygons.append(negative_polygon)
    
    return result_polygons if result_polygons else [apply_relu_to_polygon(polygon)]


def is_zero_polygon(polygon):
    """
    Check if a polygon has all z-values equal to 0 (within tolerance).
    
    Args:
        polygon: numpy array of shape (n_points, 3)
    
    Returns:
        bool: True if all z-values are approximately 0
    """
    return np.all(np.abs(polygon[:, 2]) < 1e-8)


def merge_adjacent_polygons(polygons):
    """
    Merge adjacent polygons that share edges.
    
    Args:
        polygons: List of numpy arrays representing polygons
    
    Returns:
        List of merged polygons
    """
    if len(polygons) <= 1:
        return polygons
    
    # Keep merging until no more merges are possible
    current_polygons = polygons[:]
    
    while True:
        merged_any = False
        new_polygons = []
        used = [False] * len(current_polygons)
        
        for i in range(len(current_polygons)):
            if used[i]:
                continue
                
            # Start a new merge group with polygon i
            merge_group = [current_polygons[i]]
            used[i] = True
            
            # Keep looking for polygons to add to this group
            added_to_group = True
            while added_to_group:
                added_to_group = False
                for j in range(len(current_polygons)):
                    if used[j]:
                        continue
                    
                    # Check if polygon j is adjacent to any polygon in current merge group
                    for group_poly in merge_group:
                        if polygons_share_edge(group_poly, current_polygons[j]):
                            merge_group.append(current_polygons[j])
                            used[j] = True
                            added_to_group = True
                            merged_any = True
                            break
                    
                    if added_to_group:
                        break
            
            # Merge all polygons in this group
            if len(merge_group) == 1:
                new_polygons.append(merge_group[0])
            else:
                merged_polygon = merge_polygon_group(merge_group)
                new_polygons.append(merged_polygon)
        
        current_polygons = new_polygons
        
        # If no merges happened this iteration, we're done
        if not merged_any:
            break
    
    return current_polygons


def polygons_share_edge(poly1, poly2, tolerance=1e-6):
    """
    Check if two polygons share an edge (two consecutive vertices).
    
    Args:
        poly1, poly2: numpy arrays representing polygons
        tolerance: tolerance for point comparison
    
    Returns:
        bool: True if polygons share an edge
    """
    # Get edges from both polygons
    edges1 = get_polygon_edges(poly1)
    edges2 = get_polygon_edges(poly2)
    
    # Check if any edges match (in either direction)
    for edge1 in edges1:
        for edge2 in edges2:
            # Check if edges are the same (forward or backward)
            if (edges_equal(edge1, edge2, tolerance) or 
                edges_equal(edge1, (edge2[1], edge2[0]), tolerance)):
                return True
    
    return False


def get_polygon_edges(polygon):
    """
    Get all edges of a polygon as pairs of points.
    
    Args:
        polygon: numpy array of shape (n_points, 3)
    
    Returns:
        List of (point1, point2) tuples representing edges
    """
    edges = []
    n_points = len(polygon)
    for i in range(n_points):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n_points]
        edges.append((p1, p2))
    return edges


def edges_equal(edge1, edge2, tolerance=1e-6):
    """
    Check if two edges are equal within tolerance.
    
    Args:
        edge1, edge2: tuples of (start_point, end_point)
        tolerance: tolerance for comparison
    
    Returns:
        bool: True if edges are equal
    """
    p1_start, p1_end = edge1
    p2_start, p2_end = edge2
    
    return (np.allclose(p1_start, p2_start, atol=tolerance) and 
            np.allclose(p1_end, p2_end, atol=tolerance))


def merge_polygon_group(polygons):
    """
    Merge a group of adjacent polygons into a single polygon by finding the outer boundary.
    Uses a simpler approach: collect all unique vertices and compute the boundary.
    
    Args:
        polygons: List of numpy arrays representing polygons to merge
    
    Returns:
        numpy array representing the merged polygon
    """
    if len(polygons) == 1:
        return polygons[0]
    
    # Collect all unique vertices from all polygons
    all_vertices = []
    for polygon in polygons:
        for vertex in polygon:
            # Check if this vertex is already in our list
            is_duplicate = False
            for existing in all_vertices:
                if np.allclose(vertex, existing, atol=1e-8):
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_vertices.append(vertex.copy())
    
    if len(all_vertices) < 3:
        return polygons[0]
    
    # Convert to numpy array for easier processing
    all_vertices = np.array(all_vertices)
    
    # Find the convex hull in 2D (x,y coordinates) to get boundary points
    try:
        from scipy.spatial import ConvexHull
        hull_2d = ConvexHull(all_vertices[:, :2])
        boundary_indices = hull_2d.vertices
        boundary_vertices = all_vertices[boundary_indices]
        
        # But we want the actual boundary, not just convex hull
        # So let's find the true outer boundary using a different approach
        outer_boundary = find_outer_boundary_detailed(polygons)
        if outer_boundary is not None and len(outer_boundary) >= 3:
            return outer_boundary
        else:
            # Fallback to convex hull if detailed boundary fails
            return boundary_vertices
            
    except Exception as e:
        # If scipy not available or fails, use angle-based sorting
        return sort_polygon_points_3d(all_vertices)


def find_outer_boundary_detailed(polygons):
    """
    Find the true outer boundary of merged polygons by identifying boundary edges.
    
    Args:
        polygons: List of polygon numpy arrays
        
    Returns:
        numpy array of boundary vertices, or None if failed
    """
    # Create a dictionary to count how many times each edge appears
    edge_count = {}
    edge_to_vertices = {}  # Map edges to their actual vertex objects
    
    for polygon in polygons:
        n_vertices = len(polygon)
        for i in range(n_vertices):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % n_vertices]
            
            # Create edge key using rounded coordinates for robust comparison
            def vertex_key(v):
                return (round(v[0], 8), round(v[1], 8), round(v[2], 8))
            
            key1 = vertex_key(v1)
            key2 = vertex_key(v2)
            
            # Normalize edge direction (smaller key first)
            if key1 <= key2:
                edge_key = (key1, key2)
                edge_direction = (v1, v2)
            else:
                edge_key = (key2, key1) 
                edge_direction = (v2, v1)
            
            # Count this edge
            if edge_key in edge_count:
                edge_count[edge_key] += 1
            else:
                edge_count[edge_key] = 1
                edge_to_vertices[edge_key] = edge_direction
    
    # Find boundary edges (edges that appear exactly once)
    boundary_edges = []
    for edge_key, count in edge_count.items():
        if count == 1:
            boundary_edges.append(edge_to_vertices[edge_key])
    
    if len(boundary_edges) < 3:
        return None
    
    # Build connected boundary by following edges
    if len(boundary_edges) == 0:
        return None
        
    # Start with first edge
    boundary_vertices = [boundary_edges[0][0], boundary_edges[0][1]]
    used_edges = {0}
    
    # Keep adding connected edges
    max_iterations = len(boundary_edges) * 2  # Prevent infinite loops
    iterations = 0
    
    while len(used_edges) < len(boundary_edges) and iterations < max_iterations:
        iterations += 1
        current_end = boundary_vertices[-1]
        found_connection = False
        
        for i, (start, end) in enumerate(boundary_edges):
            if i in used_edges:
                continue
            
            # Check if this edge connects to our current end
            if np.allclose(current_end, start, atol=1e-8):
                boundary_vertices.append(end)
                used_edges.add(i)
                found_connection = True
                break
            elif np.allclose(current_end, end, atol=1e-8):
                boundary_vertices.append(start)  
                used_edges.add(i)
                found_connection = True
                break
        
        if not found_connection:
            # Check if we've formed a closed loop
            if len(boundary_vertices) > 2 and np.allclose(boundary_vertices[-1], boundary_vertices[0], atol=1e-8):
                break
            else:
                # Try to continue from a different unused edge
                remaining_edges = [i for i in range(len(boundary_edges)) if i not in used_edges]
                if remaining_edges:
                    next_edge_idx = remaining_edges[0]
                    start, end = boundary_edges[next_edge_idx]
                    boundary_vertices.extend([start, end])
                    used_edges.add(next_edge_idx)
                else:
                    break
    
    # Remove duplicate points (especially the closing point)
    if len(boundary_vertices) > 1 and np.allclose(boundary_vertices[-1], boundary_vertices[0], atol=1e-8):
        boundary_vertices = boundary_vertices[:-1]
    
    # Remove any remaining duplicates
    unique_vertices = []
    for vertex in boundary_vertices:
        is_duplicate = False
        for existing in unique_vertices:
            if np.allclose(vertex, existing, atol=1e-8):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(vertex)
    
    if len(unique_vertices) >= 3:
        return np.array(unique_vertices)
    else:
        return None
    """
    Apply ReLU operation to a polygon's z-values (clip negative values to 0).
    
    Args:
        polygon: numpy array of shape (n_points, 3)
    
    Returns:
        numpy array with z-values clipped to be >= 0
    """
    clipped_polygon = polygon.copy()
    clipped_polygon[:, 2] = np.maximum(clipped_polygon[:, 2], 0.0)
    return clipped_polygon


def sort_polygon_points_3d(points):
    """
    Sort 3D polygon points to maintain proper ordering.
    Projects to 2D (x,y) plane for sorting since we're dealing with surfaces.
    
    Args:
        points: numpy array of shape (n_points, 3)
    
    Returns:
        numpy array of sorted points
    """
    if len(points) < 3:
        return points
    
    # Find centroid in x,y plane
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    
    # Calculate angles from centroid in x,y plane
    angles = np.arctan2(points[:, 1] - centroid_y, points[:, 0] - centroid_x)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]