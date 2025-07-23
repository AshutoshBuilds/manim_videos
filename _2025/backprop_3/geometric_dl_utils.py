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







def split_polygons_with_relu(layer_polygons_3d):
    """
    Split 3D polygons that cross the z=0 plane (ReLU boundary).
    
    Args:
        layer_polygons_3d: List of lists of numpy arrays representing 3D polygons
                          Each sublist represents polygons for one neuron
                          Each numpy array is a polygon with shape (n_points, 3)
    
    Returns:
        List of lists of numpy arrays with split polygons added
    """
    result = []
    
    for neuron_idx, neuron_polygons in enumerate(layer_polygons_3d):
        neuron_result = []
        
        for polygon in neuron_polygons:
            if len(polygon) < 3:
                # Skip degenerate polygons
                neuron_result.append(polygon)
                continue
                
            # Check if polygon crosses z=0
            z_values = polygon[:, 2]
            min_z = np.min(z_values)
            max_z = np.max(z_values)
            
            if min_z >= 0 or max_z <= 0:
                # Polygon doesn't cross z=0, keep as is
                neuron_result.append(polygon)
            else:
                # Polygon crosses z=0, need to split it
                split_polygons = split_polygon_at_z_zero(polygon)
                neuron_result.extend(split_polygons)
        
        result.append(neuron_result)
    
    return result


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
    
    # Convert to numpy arrays and sort points to maintain proper polygon order
    result_polygons = []
    
    if len(positive_polygon) >= 3:
        positive_polygon = np.array(positive_polygon)
        positive_polygon = sort_polygon_points_3d(positive_polygon)
        result_polygons.append(positive_polygon)
    
    if len(negative_polygon) >= 3:
        negative_polygon = np.array(negative_polygon)
        negative_polygon = sort_polygon_points_3d(negative_polygon)
        result_polygons.append(negative_polygon)
    
    return result_polygons if result_polygons else [polygon]


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

# def get_relu_joint(weight_1, weight_2, bias, extent=1):
#     if np.abs(weight_2) < 1e-8: 
#         x_intercept = -bias / weight_1
#         return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
#     elif np.abs(weight_1) < 1e-8:
#         y_intercept = -bias / weight_2
#         return [[-extent, y_intercept], [extent, y_intercept]] if -extent <= y_intercept <= extent else []
#     else:
#         points = []
#         for x in [-extent, extent]:
#             y = (-x * weight_1 - bias) / weight_2
#             if -extent <= y <= extent: points.append([x, y])
#         for y in [-extent, extent]:
#             x = (-y * weight_2 - bias) / weight_1
#             if -extent <= x <= extent: points.append([x, y])
#         unique_points = []
#         for p in points:
#             is_duplicate = False
#             for existing in unique_points:
#                 if abs(p[0] - existing[0]) < 1e-8 and abs(p[1] - existing[1]) < 1e-8:
#                     is_duplicate = True
#                     break
#             if not is_duplicate:
#                 unique_points.append(p)
#         return unique_points

# def line_from_joint_points(joint_points):
#     if joint_points:
#         # Create 3D points for the joint line
#         joint_3d_points = []
#         for point in joint_points:
#             x, y = point
#             z = 0
#             joint_3d_points.append([x, y, z])
        
#         if len(joint_3d_points) >= 2:
#             joint_line = DashedLine(
#                 start=[joint_points[0][0], joint_points[0][1], 0],
#                 end=[joint_points[1][0], joint_points[1][1], 0],
#                 color=WHITE,
#                 stroke_width=3,
#                 dash_length=0.05
#             )
#             return joint_line


