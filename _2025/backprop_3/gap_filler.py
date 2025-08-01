import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

def fill_gaps(polygons, indicator):
    """
    Takes in list of numpy array of polygon coordinates and boolean indicator list 
    of the same length. Looks for gaps in polygons on the [-1,1] plane, and fills 
    with new polygons, using nearest neighbors to pick choose the right value for 
    indicator (will be 0 or 1). Returns a new longer list of polygons and longer 
    indicator list.
    
    Args:
        polygons: List of numpy arrays, each of shape (n, 3) representing polygon vertices
        indicator: List of 0s and 1s indicating which surface each polygon belongs to
    
    Returns:
        filled_polygons: Extended list of polygons including gap-filling polygons
        filled_indicator: Extended indicator list
    """
    # Convert to shapely polygons (using only x,y coordinates)
    shapely_polys = []
    valid_indices = []
    
    for i, poly in enumerate(polygons):
        try:
            # Use only x,y coordinates
            poly_2d = Polygon(poly[:, :2])
            
            # Make valid if needed
            if not poly_2d.is_valid:
                poly_2d = make_valid(poly_2d)
            
            if poly_2d.is_valid and poly_2d.area > 1e-10:
                shapely_polys.append(poly_2d)
                valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Could not process polygon {i}: {e}")
            continue
    
    if not shapely_polys:
        return polygons, indicator
    
    # Create the bounding box for the [-1, 1] x [-1, 1] domain
    domain = box(-1, -1, 1, 1)
    
    # Union all existing polygons
    covered_area = unary_union(shapely_polys)
    
    # Find gaps
    gaps = domain.difference(covered_area)
    
    # Convert gaps to list of polygons
    gap_polygons = []
    if hasattr(gaps, 'geoms'):
        # MultiPolygon case
        gap_polygons = [g for g in gaps.geoms if g.area > 1e-10]
    elif isinstance(gaps, Polygon) and gaps.area > 1e-10:
        # Single polygon case
        gap_polygons = [gaps]
    
    # If no gaps found, return original
    if not gap_polygons:
        return polygons, indicator
    
    # Fill gaps by assigning indicators based on nearest neighbors
    filled_polygons = list(polygons)
    filled_indicator = list(indicator)
    
    for gap in gap_polygons:
        # Get centroid of gap
        gap_centroid = np.array([gap.centroid.x, gap.centroid.y])
        
        # Find nearest existing polygon
        min_dist = float('inf')
        nearest_idx = 0
        
        for idx in valid_indices:
            poly = polygons[idx]
            # Compute distance from gap centroid to polygon centroid
            poly_centroid = np.mean(poly[:, :2], axis=0)
            dist = np.linalg.norm(gap_centroid - poly_centroid)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        # Create 3D polygon for the gap
        # Extract vertices from shapely polygon
        gap_coords = np.array(gap.exterior.coords[:-1])  # Remove duplicate last point
        
        # Add z-coordinate by interpolating from nearest polygon
        # For simplicity, use the mean z-value of the nearest polygon
        nearest_z = np.mean(polygons[nearest_idx][:, 2])
        
        gap_3d = np.zeros((len(gap_coords), 3))
        gap_3d[:, :2] = gap_coords
        gap_3d[:, 2] = nearest_z
        
        # Add to results
        filled_polygons.append(gap_3d)
        filled_indicator.append(indicator[nearest_idx])
    
    return filled_polygons, filled_indicator


def fill_gaps_advanced(polygons, indicator, k_neighbors=5):
    """
    Advanced version that uses k-nearest neighbors for better indicator assignment
    and interpolates z-values more accurately.
    """
    # Convert to shapely polygons (using only x,y coordinates)
    shapely_polys = []
    valid_indices = []
    
    for i, poly in enumerate(polygons):
        try:
            # Use only x,y coordinates
            poly_2d = Polygon(poly[:, :2])
            
            # Make valid if needed
            if not poly_2d.is_valid:
                poly_2d = make_valid(poly_2d)
            
            if poly_2d.is_valid and poly_2d.area > 1e-10:
                shapely_polys.append(poly_2d)
                valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Could not process polygon {i}: {e}")
            continue
    
    if not shapely_polys:
        return polygons, indicator
    
    # Create the bounding box for the [-1, 1] x [-1, 1] domain
    domain = box(-1, -1, 1, 1)
    
    # Union all existing polygons
    covered_area = unary_union(shapely_polys)
    
    # Find gaps
    gaps = domain.difference(covered_area)
    
    # Convert gaps to list of polygons
    gap_polygons = []
    if hasattr(gaps, 'geoms'):
        # MultiPolygon case
        gap_polygons = [g for g in gaps.geoms if g.area > 1e-10]
    elif isinstance(gaps, Polygon) and gaps.area > 1e-10:
        # Single polygon case
        gap_polygons = [gaps]
    
    # If no gaps found, return original
    if not gap_polygons:
        return polygons, indicator
    
    print(f"Found {len(gap_polygons)} gaps to fill")
    
    # Fill gaps by assigning indicators based on k-nearest neighbors
    filled_polygons = list(polygons)
    filled_indicator = list(indicator)
    
    for gap in gap_polygons:
        # Get centroid of gap
        gap_centroid = np.array([gap.centroid.x, gap.centroid.y])
        
        # Find k nearest existing polygons
        distances = []
        for idx in valid_indices:
            poly = polygons[idx]
            poly_centroid = np.mean(poly[:, :2], axis=0)
            dist = np.linalg.norm(gap_centroid - poly_centroid)
            distances.append((dist, idx))
        
        # Sort by distance and take k nearest
        distances.sort()
        k_nearest = distances[:min(k_neighbors, len(distances))]
        
        # Vote on indicator (majority wins)
        votes = [indicator[idx] for _, idx in k_nearest]
        gap_indicator = 1 if sum(votes) > len(votes) / 2 else 0
        
        # Create 3D polygon for the gap
        gap_coords = np.array(gap.exterior.coords[:-1])  # Remove duplicate last point
        
        # Interpolate z-values from k nearest neighbors
        # Weight by inverse distance
        total_weight = 0
        weighted_z = 0
        
        for dist, idx in k_nearest:
            weight = 1.0 / (dist + 1e-6)  # Add small epsilon to avoid division by zero
            total_weight += weight
            weighted_z += weight * np.mean(polygons[idx][:, 2])
        
        interpolated_z = weighted_z / total_weight if total_weight > 0 else 0
        
        gap_3d = np.zeros((len(gap_coords), 3))
        gap_3d[:, :2] = gap_coords
        gap_3d[:, 2] = interpolated_z
        
        # Add to results
        filled_polygons.append(gap_3d)
        filled_indicator.append(gap_indicator)
    
    return filled_polygons, filled_indicator