import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def compute_polygon_features(polygon):
    """
    Compute centroid and area of a polygon.
    
    Args:
        polygon: Nx2 or Nx3 numpy array of vertices (only first 2 columns used)
    
    Returns:
        tuple: (centroid, area)
    """
    # Use only first 2 columns (x, y coordinates)
    vertices = polygon[:, :2]
    
    # Compute centroid
    centroid = np.mean(vertices, axis=0)
    
    # Compute area using shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    
    return centroid, area

def compute_polygon_distance(poly1, poly2, centroid_weight=1.0, area_weight=0.1):
    """
    Compute distance between two polygons based on centroid and area.
    
    Args:
        poly1, poly2: Polygon vertex arrays
        centroid_weight: Weight for centroid distance
        area_weight: Weight for area difference
    
    Returns:
        float: Combined distance metric
    """
    centroid1, area1 = compute_polygon_features(poly1)
    centroid2, area2 = compute_polygon_features(poly2)
    
    # Euclidean distance between centroids
    centroid_dist = np.linalg.norm(centroid1 - centroid2)
    
    # Normalized area difference
    area_diff = abs(area1 - area2) / max(area1, area2, 1e-8)
    
    return centroid_weight * centroid_dist + area_weight * area_diff

def reorder_polygons_greedy(prev_polygons, curr_polygons, 
                          centroid_weight=1.0, area_weight=0.1):
    """
    Reorder current polygons to best match previous polygons using greedy assignment.
    
    Args:
        prev_polygons: List of polygon arrays from previous timestep
        curr_polygons: List of polygon arrays from current timestep
        centroid_weight: Weight for centroid distance in matching
        area_weight: Weight for area difference in matching
    
    Returns:
        list: Reordered current polygons
    """
    if not prev_polygons or not curr_polygons:
        return curr_polygons
    
    n_prev = len(prev_polygons)
    n_curr = len(curr_polygons)
    
    # Compute distance matrix
    distances = np.zeros((n_prev, n_curr))
    for i, prev_poly in enumerate(prev_polygons):
        for j, curr_poly in enumerate(curr_polygons):
            distances[i, j] = compute_polygon_distance(
                prev_poly, curr_poly, centroid_weight, area_weight
            )
    
    # Greedy assignment: for each previous polygon, find closest current polygon
    used_indices = set()
    reordered = []
    
    for i in range(n_prev):
        # Find closest unused current polygon
        best_j = None
        best_dist = float('inf')
        
        for j in range(n_curr):
            if j not in used_indices and distances[i, j] < best_dist:
                best_dist = distances[i, j]
                best_j = j
        
        if best_j is not None:
            reordered.append(curr_polygons[best_j])
            used_indices.add(best_j)
        else:
            # No unused polygon found, duplicate the closest one
            best_j = np.argmin(distances[i, :])
            reordered.append(curr_polygons[best_j])
    
    # Add any unmatched current polygons at the end
    for j in range(n_curr):
        if j not in used_indices:
            reordered.append(curr_polygons[j])
    
    return reordered

def reorder_polygons_optimal(prev_polygons, curr_polygons, 
                           centroid_weight=1.0, area_weight=0.1):
    """
    Reorder current polygons using optimal assignment (Hungarian algorithm).
    This gives better results but is O(n³) vs O(n²) for greedy.
    
    Args:
        prev_polygons: List of polygon arrays from previous timestep
        curr_polygons: List of polygon arrays from current timestep
        centroid_weight: Weight for centroid distance in matching
        area_weight: Weight for area difference in matching
    
    Returns:
        list: Reordered current polygons
    """
    if not prev_polygons or not curr_polygons:
        return curr_polygons
    
    n_prev = len(prev_polygons)
    n_curr = len(curr_polygons)
    
    # Compute distance matrix
    distances = np.zeros((n_prev, n_curr))
    for i, prev_poly in enumerate(prev_polygons):
        for j, curr_poly in enumerate(curr_polygons):
            distances[i, j] = compute_polygon_distance(
                prev_poly, curr_poly, centroid_weight, area_weight
            )
    
    # Handle case where number of polygons changed
    if n_prev <= n_curr:
        # More or equal current polygons
        row_ind, col_ind = linear_sum_assignment(distances)
        reordered = [None] * n_prev
        used_indices = set()
        
        for i, j in zip(row_ind, col_ind):
            reordered[i] = curr_polygons[j]
            used_indices.add(j)
        
        # Add unmatched polygons at the end
        for j in range(n_curr):
            if j not in used_indices:
                reordered.append(curr_polygons[j])
    else:
        # Fewer current polygons than previous
        # Create extended cost matrix
        extended_distances = np.full((n_prev, n_prev), np.max(distances) * 2)
        extended_distances[:n_prev, :n_curr] = distances
        
        row_ind, col_ind = linear_sum_assignment(extended_distances)
        reordered = []
        
        for i, j in zip(row_ind, col_ind):
            if j < n_curr:  # Valid assignment
                reordered.append(curr_polygons[j])
    
    return reordered

# Example usage and test function
def test_reordering():
    """Test the reordering functions with simple example polygons."""
    
    # Create some test polygons (squares at different positions)
    def make_square(center_x, center_y, size=0.5):
        return np.array([
            [center_x - size/2, center_y - size/2],
            [center_x + size/2, center_y - size/2],
            [center_x + size/2, center_y + size/2],
            [center_x - size/2, center_y + size/2]
        ])
    
    # Previous timestep polygons
    prev_polygons = [
        make_square(0, 0),      # polygon 0
        make_square(1, 0),      # polygon 1  
        make_square(0, 1),      # polygon 2
    ]
    
    # Current timestep polygons (slightly moved and reordered)
    curr_polygons = [
        make_square(0.1, 1.1),  # moved version of polygon 2
        make_square(1.1, 0.1),  # moved version of polygon 1
        make_square(0.1, 0.1),  # moved version of polygon 0
        make_square(2, 2),      # new polygon
    ]
    
    print("Original order - centroids:")
    for i, poly in enumerate(curr_polygons):
        centroid, _ = compute_polygon_features(poly)
        print(f"  Polygon {i}: {centroid}")
    
    # Test greedy reordering
    reordered_greedy = reorder_polygons_greedy(prev_polygons, curr_polygons)
    print("\nGreedy reordered - centroids:")
    for i, poly in enumerate(reordered_greedy):
        centroid, _ = compute_polygon_features(poly)
        print(f"  Polygon {i}: {centroid}")
    
    # Test optimal reordering
    reordered_optimal = reorder_polygons_optimal(prev_polygons, curr_polygons)
    print("\nOptimal reordered - centroids:")
    for i, poly in enumerate(reordered_optimal):
        centroid, _ = compute_polygon_features(poly)
        print(f"  Polygon {i}: {centroid}")

if __name__ == "__main__":
    test_reordering()