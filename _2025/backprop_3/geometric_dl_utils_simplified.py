import torch
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely.affinity
import copy
from tqdm import tqdm


def compute_top_polytope(model, tiling_2d):
    hyperspace_polygons=process_with_layers(model.model, tiling_2d)

    my_top_polygons=[]
    my_indicator=[]
    for p1, p2 in zip(hyperspace_polygons[0], hyperspace_polygons[1]):
        if np.all(p1[:,2]>p2[:,2]):
            my_top_polygons.append(p1)
            my_indicator.append(0)
        elif np.all(p2[:,2]>p1[:,2]):
            my_top_polygons.append(p2)
            my_indicator.append(1)
        elif np.max(p1[:,2])>np.max(p2[:,2]): #Is this crazy? Seems like it works? I think i only need these last 2 really. 
            my_top_polygons.append(p1)
            my_indicator.append(0)
        elif np.max(p2[:,2])>np.max(p1[:,2]): #Is this crazy?
            my_top_polygons.append(p2)
            my_indicator.append(1)
        else:
            #Tie break - GPT idea
            my_top_polygons.append(p1)   
            my_indicator.append(0)
    return my_indicator, my_top_polygons


def process_with_layers(model_layers, polygons_flat):
    '''
    model_layers is a pytorch Sequential object, in practice is typically a subset of a model, or could be full model. 
    polygons_flat is a list of 3d or 2d polygons as Nx3 or Nx2 numpy arrays.
    In the Nx3 case, process_with_linear_layer only uses first xy coords, not the z value
    Returns a new list of list of polygons. Outer dimension is over neurons, inner list is over polygons as Nx3 numpy arrays
    The xy values of each polygon are the same across all neurons
    The z value of each point corresponds to the output of that neuron for that vertex
    '''

    
    # Get number of neurons from the model's output dimension
    with torch.no_grad():
        # Use first polygon to determine output size
        test_input = torch.tensor(polygons_flat[0][:,:2]).float()
        test_output = model_layers(test_input)
        num_neurons = test_output.shape[1]
    
    # Initialize result: list of lists, outer dim = neurons, inner dim = polygons
    result = [[] for _ in range(num_neurons)]
    
    for i, p in enumerate(polygons_flat):
        with torch.no_grad():
            out = model_layers(torch.tensor(p[:,:2]).float())  # Returns num_vertices x num_neurons
            
            # For each neuron, create a new polygon with xy from original and z from neuron output
            for neuron_idx in range(num_neurons):
                # Create Nx3 array: xy from original polygon, z from neuron output
                new_polygon = np.zeros((p.shape[0], 3))
                new_polygon[:, :2] = p[:, :2]  # Copy xy coordinates
                new_polygon[:, 2] = out[:, neuron_idx].numpy()  # Set z to neuron output
                
                result[neuron_idx].append(new_polygon)
    
    return result


def clip_polygons(polygons):
    '''
    Clip polygon z values to zero for various depths of lists
    Im sure theres a cool recursive solution
    '''
    clipped_polygons=copy.deepcopy(polygons)
    for l1 in clipped_polygons:
        if isinstance(l1, np.ndarray): l1[:,2]=np.maximum(0,  l1[:,2])
        else: 
            for l2 in l1: 
                if isinstance(l2, np.ndarray): l2[:,2]=np.maximum(0,  l2[:,2])
                else:
                    for l3 in l2: 
                        if isinstance(l3, np.ndarray): l3[:,2]=np.maximum(0,  l3[:,2])
                        
    return clipped_polygons

def split_polygons_with_relu_simple(polygons):
    """
    Split 3D polygons that cross the z=0 plane (ReLU boundary)
    If a polygon does not cross z=0, simply pass through. If a polygon does cross z=0, split at the z=0 line, remove old polygon and
    add new polygons. 
    
    Args:
        polygons: List of lists of numpy arrays representing 3D polygons
                          Each sublist represents polygons for one neuron
                          Each numpy array is a polygon with shape (n_points, 3)
    
    Returns:
        - split_polygons: List of lists of lists of numpy arrays 
                         First two dimensions match input structure
                         Third dimension is length 1 for unsplit polygons, length 2 for split polygons
    """
    
    def intersect_edge_with_z_plane(p1, p2, z=0):
        """Find intersection of edge p1-p2 with z=z plane"""
        if abs(p1[2] - p2[2]) < 1e-10:  # Edge is parallel to z plane
            return None
        
        # Parametric line: point = p1 + t*(p2-p1)
        # At intersection: p1[2] + t*(p2[2]-p1[2]) = z
        t = (z - p1[2]) / (p2[2] - p1[2])
        
        if 0 <= t <= 1:  # Intersection is within the edge
            intersection = p1 + t * (p2 - p1)
            intersection[2] = z  # Ensure exact z value
            return intersection
        return None
    
    def split_polygon_at_z_plane(polygon, z=0):
        """Split a polygon at the z=z plane"""
        points = polygon
        n = len(points)
        
        # Check if polygon crosses the plane
        z_values = points[:, 2]
        if np.all(z_values >= z) or np.all(z_values <= z):
            # Polygon doesn't cross the plane
            return [polygon]
        
        # Find intersections and classify points
        above_points = []
        below_points = []
        intersection_points = []
        
        for i in range(n):
            curr_point = points[i]
            next_point = points[(i + 1) % n]
            
            # Add current point to appropriate list
            if curr_point[2] >= z:
                above_points.append(curr_point.copy())
            if curr_point[2] <= z:
                below_points.append(curr_point.copy())
            
            # Check for intersection with next edge
            intersection = intersect_edge_with_z_plane(curr_point, next_point, z)
            if intersection is not None:
                intersection_points.append(intersection)
                # Add intersection to both polygons
                above_points.append(intersection.copy())
                below_points.append(intersection.copy())
        
        # Create the split polygons
        result_polygons = []
        
        if len(above_points) >= 3:
            result_polygons.append(np.array(above_points))
        
        if len(below_points) >= 3:
            result_polygons.append(np.array(below_points))
        
        return result_polygons if result_polygons else [polygon]
    
    # Process each neuron's polygons
    split_polygons = []
    
    for neuron_polygons in polygons:
        neuron_split_polygons = []
        
        for polygon in neuron_polygons:
            # Split this polygon at z=0
            split_parts = split_polygon_at_z_plane(polygon, z=0)
            
            # Wrap result in a list to maintain triple-nested structure
            # Length 1 for unsplit, length 2 for split polygons
            neuron_split_polygons.append(split_parts)
        
        split_polygons.append(neuron_split_polygons)
    
    return split_polygons


def recompute_tiling(polygons_nested, min_area=1e-10):
    '''
    polygons_nested is a list of list of list of polygons in 3d space as Nx3 or Nx2 numpy arrays 
    this method only uses the first 2 dimensions
    The outer list correspond to neurons in a given layer, recompute_tiling collapses across this dimension
    The second layer of list correspond to "input polygons" into the layer that maybe have been split by the given layer's Relu
    The inner layer correspond to resulting polygons from splitting a polygon that was input into this layer
    #The length of the second layer lists shoudl all the same
    recompute_tiling iterates through input polygons (second layer list). 
    For each input polygon, recompute_tiling collapses this polygon across all neurons by finding the intersections of all new polygons
    generate withing this input polygon. In the null case, all polygons for a specific neuron will be lists of length 1, where the polygon 
    has not been split by any of the neurons - in this case recompute_tiling just returns the original polygon borders
    If only one neuron has split the polygon at hand, then polygons_nested can just return these 2 new polygons
    Otherwise recompute_tiling needs to recompute N new polygons based on the intersections of the new polygons formed by mutiple neurons
    recompute_tiling returns a list of list of polygons, where the outer list correspond to the number of input polygons (should match the length
    of the second nested list passed in), and the inner lists are all the polygons that an input polygon are split into. 
    '''
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import shapely.affinity
    
    def numpy_to_shapely(poly_array):
        """Convert numpy array to shapely polygon using only xy coordinates"""
        return Polygon(poly_array[:, :2])
    
    def shapely_to_numpy(shapely_poly, z_value=0):
        """Convert shapely polygon back to numpy array with z coordinate"""
        coords = list(shapely_poly.exterior.coords)[:-1]  # Remove duplicate last point
        result = np.zeros((len(coords), 3))
        result[:, :2] = coords
        result[:, 2] = z_value
        return result
    
    def find_polygon_intersections(polygon_lists):
        """
        Find all distinct regions created by intersecting multiple sets of polygons
        
        Args:
            polygon_lists: List of lists, where each inner list contains shapely polygons
                          from one neuron's splitting of the original polygon
        
        Returns:
            List of shapely polygons representing all distinct regions
        """
        if not polygon_lists or all(len(plist) == 0 for plist in polygon_lists):
            return []
        
        # Start with the first neuron's polygons
        current_regions = polygon_lists[0].copy()
        
        # Intersect with each subsequent neuron's polygons
        for neuron_polys in polygon_lists[1:]:
            new_regions = []
            
            for current_poly in current_regions:
                for neuron_poly in neuron_polys:
                    intersection = current_poly.intersection(neuron_poly)
                    
                    # Handle different intersection types
                    if intersection.is_empty:
                        continue
                    elif hasattr(intersection, 'geoms'):  # MultiPolygon
                        for geom in intersection.geoms:
                            if isinstance(geom, Polygon) and geom.area > min_area:
                                new_regions.append(geom)
                    elif isinstance(intersection, Polygon) and intersection.area > min_area:
                        new_regions.append(intersection)
            
            current_regions = new_regions
            
            # If no regions remain, break early
            if not current_regions:
                break
        
        return current_regions
    
    # Get dimensions
    if not polygons_nested or not polygons_nested[0]:
        return []
    
    num_neurons = len(polygons_nested)
    num_input_polygons = len(polygons_nested[0])
    
    # Result: list of lists, outer = input polygons, inner = resulting split polygons
    result = []
    
    # Process each input polygon
    for input_poly_idx in range(num_input_polygons):
        # Collect all split polygons for this input polygon across all neurons
        neuron_polygon_lists = []
        
        for neuron_idx in range(num_neurons):
            # Get the split polygons for this input polygon from this neuron
            split_polys = polygons_nested[neuron_idx][input_poly_idx]
            
            # Convert to shapely polygons
            shapely_polys = [numpy_to_shapely(poly) for poly in split_polys]
            neuron_polygon_lists.append(shapely_polys)
        
        # Check how many neurons actually split this polygon
        split_counts = [len(plist) for plist in neuron_polygon_lists]
        num_splits = sum(1 for count in split_counts if count > 1)
        
        if num_splits == 0:
            # No neuron split this polygon, return original
            original_poly = polygons_nested[0][input_poly_idx][0]  # Get from first neuron
            result.append([original_poly])
            
        elif num_splits == 1:
            # Only one neuron split this polygon, return its splits
            for plist in neuron_polygon_lists:
                if len(plist) > 1:
                    # Convert back to numpy arrays
                    numpy_splits = [shapely_to_numpy(sp) for sp in plist]
                    result.append(numpy_splits)
                    break
        else:
            # Multiple neurons split this polygon, need to find intersections
            intersected_regions = find_polygon_intersections(neuron_polygon_lists)
            
            if intersected_regions:
                # Convert back to numpy arrays
                numpy_regions = [shapely_to_numpy(region) for region in intersected_regions]
                result.append(numpy_regions)
            else:
                # Fallback: return original polygon if intersection fails
                original_poly = polygons_nested[0][input_poly_idx][0]
                result.append([original_poly])
    
    return result


## --- Merging z=0 polygons --- ##

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union, snap

def merge_zero_regions(polygons,
                       snap_tol: float = 1e-8,
                       buffer_eps: float = 1e-6,
                       min_area: float = 1e-8):
    """
    Merge all adjacent or point‑touching z=0 polygons, for:
      • Flat list of Nx3 arrays     → returns List[Nx3]
      • Per‑neuron flat lists       → returns List[List[Nx3]]
      • Per‑neuron nested lists     → returns List[List[Nx3]]

    Polygons with any nonzero z remain untouched.

    Args:
      polygons: one of
        - List[np.ndarray]  
        - List[List[np.ndarray]]  
        - List[List[List[np.ndarray]]]
      snap_tol: max distance to snap nearby vertices/edges
      buffer_eps: small buffer to bridge point‑contacts
      min_area: discard merged pieces smaller than this

    Returns:
      Either a flat List[np.ndarray] (if input was flat),
      or a List[List[np.ndarray]] of merged per‑neuron lists.
    """

    def process_flat(flat_list):
        # 1) split zero‑height vs non‑zero
        zero = [p for p in flat_list if np.allclose(p[:,2], 0)]
        nonz = [p for p in flat_list if not np.allclose(p[:,2], 0)]

        merged = []
        if zero:
            # to shapely
            shapes = [Polygon(p[:, :2]) for p in zero]
            # snap → align tiny gaps
            uni = unary_union(shapes)
            snapped = [snap(s, uni, snap_tol) for s in shapes]
            # buffer out & in → fuse point‑contacts
            buff = [s.buffer(buffer_eps, join_style=2) for s in snapped]
            uni2 = unary_union(buff).buffer(-buffer_eps, join_style=2)

            geoms = uni2.geoms if hasattr(uni2, "geoms") else [uni2]
            for g in geoms:
                if isinstance(g, Polygon) and g.area >= min_area:
                    coords = list(g.exterior.coords)[:-1]
                    arr = np.zeros((len(coords), 3), dtype=float)
                    arr[:, :2] = coords
                    # z stays zero
                    merged.append(arr)

        # combine merged zero regions + original non‑zero
        return merged + nonz

    # --- Detect flat list of np.ndarray (case #1) ---
    if isinstance(polygons, list) and all(isinstance(p, np.ndarray) for p in polygons):
        return process_flat(polygons)

    # --- Otherwise assume per‑neuron list ---
    out = []
    for neuron in polygons:
        if not neuron:
            out.append([])
            continue

        # detect if this neuron entry is flat list or nested two levels
        first = neuron[0]
        if isinstance(first, np.ndarray):
            flat = neuron
        else:
            # nested: flatten one more level
            flat = [poly for group in neuron for poly in group]

        out.append(process_flat(flat))

    return out



import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def recompute_tiling_general(polygon_list, min_area=1e-10):
    """
    Given multiple tilings of the [-1,1]² plane (one list of polygons per neuron),
    compute their overlay (intersection) to produce a new single tiling.

    Args:
      polygon_list: List of length N_neurons, where each element is itself
                    a list of polygons; each polygon is an (n_pts×2) or
                    (n_pts×3) numpy array.  Only the first two cols (XY)
                    are used.
      min_area:     drop any tiny slivers below this area.

    Returns:
      new_tiling:   Flat list of M polygons, each an (m_pts×3) array
                    with cols [x, y, 0].
    """
    # 1) convert each neuron's arrays to shapely Polygons
    tilings = []
    for neuron_polys in polygon_list:
        shapely_polys = []
        for p in neuron_polys:
            # take only x,y coords
            coords = p[:, :2]
            poly = Polygon(coords)
            if poly.is_valid and poly.area > min_area:
                shapely_polys.append(poly)
        tilings.append(shapely_polys)

    if not tilings:
        return []

    # 2) start with the first neuron's tiling
    current = tilings[0]

    # 3) iteratively intersect with each subsequent neuron's tiling
    print('Retiling plane...')
    for next_tiling in tqdm(tilings[1:]):
        new_current = []
        for region in current:
            for tile in next_tiling:
                inter = region.intersection(tile)
                if inter.is_empty:
                    continue

                # handle both Polygon and MultiPolygon
                geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
                for g in geoms:
                    if isinstance(g, Polygon) and g.area > min_area:
                        new_current.append(g)

        current = new_current
        if not current:
            break

    # 4) convert final shapely Polygons back to Nx3 numpy (z=0)
    result = []
    for poly in current:
        pts = list(poly.exterior.coords)[:-1]  # drop duplicate
        arr = np.zeros((len(pts), 3), dtype=float)
        arr[:, :2] = pts
        # arr[:, 2] stays zero
        result.append(arr)

    return result



def filter_small_polygons(polygons, min_area=1e-6):
    """
    Remove any polygon whose 2D area is below `min_area`.

    Args:
      polygons: list of numpy arrays of shape (n_pts, 2) or (n_pts, 3).
                Only the first two columns (x,y) are used to compute area.
      min_area: float. Polygons with area < min_area are discarded.

    Returns:
      filtered: a new list containing only those arrays whose area >= min_area.
    """
    filtered = []
    for p in polygons:
        # take xy coords
        coords = p[:, :2]
        poly = Polygon(coords)
        if poly.is_valid and poly.area >= min_area:
            filtered.append(p)
    return filtered


## Works great in the nested case! ##
# from shapely.geometry import Polygon
# from shapely.ops import unary_union, snap
# import numpy as np

# def merge_zero_regions_nested(nested_polygons, 
#                               snap_tol=1e-8, 
#                               buffer_eps=1e-6, 
#                               min_area=1e-8):
#     """
#     Merge adjacent or corner‐touching zero‐height regions for each neuron,
#     eliminating tiny slivers and ensuring that any polygons that share
#     even a point get unioned.

#     Args:
#       nested_polygons: list of per‐neuron data, where each neuron entry is
#                        a list of lists of Nx3 numpy arrays.
#       snap_tol: maximum snapping distance to align shared edges
#       buffer_eps: small buffer distance used to bridge point‐contacts
#       min_area: drop any merged region whose area falls below this

#     Returns:
#       merged_per_neuron: list of per‐neuron flat lists of Nx3 numpy arrays.
#                          All contiguous zero‐regions have been unioned,
#                          tiny slivers removed, non‐zero polygons untouched.
#     """
#     merged_per_neuron = []

#     for neuron_polys in nested_polygons:
#         # 1) flatten two layers of grouping
#         flat = [poly
#                 for group in neuron_polys
#                 for poly in group]

#         # 2) split zero vs nonzero
#         zero_polys = [p for p in flat if np.allclose(p[:,2], 0)]
#         nonzero_polys = [p for p in flat if not np.allclose(p[:,2], 0)]

#         merged_zero_numpy = []
#         if zero_polys:
#             # convert to shapely, using only XY
#             shapely_zero = [Polygon(p[:, :2]) for p in zero_polys]

#             # 3a) snap each polygon to itself & to the others to align edges
#             #    this ensures shared edges are exactly identical
#             snapped = []
#             for poly in shapely_zero:
#                 # snap against the union of all others
#                 others = unary_union([q for q in shapely_zero if q is not poly])
#                 snapped_poly = snap(poly, others, snap_tol)
#                 snapped.append(snapped_poly)

#             # 3b) buffer out and back in to fuse any point‐contacts
#             buffered_out = [p.buffer(buffer_eps, join_style=2) for p in snapped]
#             unioned = unary_union(buffered_out)
#             cleaned = unioned.buffer(-buffer_eps, join_style=2)

#             # 4) extract resulting polygons, drop tiny ones
#             geoms = (cleaned.geoms 
#                      if hasattr(cleaned, "geoms") 
#                      else [cleaned])
#             for geom in geoms:
#                 if geom.area >= min_area and isinstance(geom, Polygon):
#                     coords = list(geom.exterior.coords)[:-1]
#                     arr = np.zeros((len(coords), 3), dtype=float)
#                     arr[:, :2] = coords
#                     # z stays zero
#                     merged_zero_numpy.append(arr)

#         # 5) combine merged zero‐regions + original nonzeros
#         merged_per_neuron.append( merged_zero_numpy + nonzero_polys )

#     return merged_per_neuron







