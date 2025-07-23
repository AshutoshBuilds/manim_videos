from manimlib import *
from functools import partial

 
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/'

from itertools import combinations, product

def get_activation_regions(w1, b1, extent=1):
    """
    Find all regions defined by first layer ReLU activations.
    Returns a list of regions, each defined by which neurons are active.
    """
    regions = []
    
    # For 2 neurons, we have 4 possible activation patterns
    for neuron1_active, neuron2_active in product([False, True], repeat=2):
        region = {
            'pattern': (neuron1_active, neuron2_active),
            'constraints': []
        }
        
        # Add constraints for this region
        if neuron1_active:
            # w1[0,0]*x + w1[0,1]*y + b1[0] >= 0
            region['constraints'].append((w1[0,0], w1[0,1], b1[0], '>='))
        else:
            # w1[0,0]*x + w1[0,1]*y + b1[0] <= 0  
            region['constraints'].append((w1[0,0], w1[0,1], b1[0], '<='))
            
        if neuron2_active:
            # w1[1,0]*x + w1[1,1]*y + b1[1] >= 0
            region['constraints'].append((w1[1,0], w1[1,1], b1[1], '>='))
        else:
            # w1[1,0]*x + w1[1,1]*y + b1[1] <= 0
            region['constraints'].append((w1[1,0], w1[1,1], b1[1], '<='))
            
        regions.append(region)
    
    return regions

def find_region_vertices(constraints, extent=1):
    """
    Find vertices of a region defined by linear constraints within [-extent, extent]^2.
    """
    # Boundary lines of the square domain
    boundary_constraints = [
        (1, 0, extent, '<='),    # x <= extent
        (-1, 0, extent, '<='),   # x >= -extent  
        (0, 1, extent, '<='),    # y <= extent
        (0, -1, extent, '<=')    # y >= -extent
    ]
    
    all_constraints = constraints + boundary_constraints
    vertices = []
    
    # Find intersections of all pairs of constraint lines
    for i in range(len(all_constraints)):
        for j in range(i+1, len(all_constraints)):
            c1 = all_constraints[i]
            c2 = all_constraints[j]
            
            # Solve system: c1[0]*x + c1[1]*y = -c1[2] and c2[0]*x + c2[1]*y = -c2[2]
            A = np.array([[c1[0], c1[1]], [c2[0], c2[1]]])
            b = np.array([-c1[2], -c2[2]])
            
            try:
                if np.abs(np.linalg.det(A)) > 1e-10:  # Non-parallel lines
                    vertex = np.linalg.solve(A, b)
                    x, y = vertex[0], vertex[1]
                    
                    # Check if vertex satisfies all constraints
                    valid = True
                    for constraint in all_constraints:
                        w1, w2, bias, op = constraint
                        value = w1 * x + w2 * y + bias
                        if op == '<=' and value > 1e-10:
                            valid = False
                            break
                        elif op == '>=' and value < -1e-10:
                            valid = False
                            break
                    
                    if valid and abs(x) <= extent + 1e-10 and abs(y) <= extent + 1e-10:
                        vertices.append([x, y])
            except np.linalg.LinAlgError:
                continue
    
    # Remove duplicate vertices
    unique_vertices = []
    for vertex in vertices:
        is_duplicate = False
        for existing in unique_vertices:
            if np.linalg.norm(np.array(vertex) - np.array(existing)) < 1e-8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(vertex)
    
    # Sort vertices in counter-clockwise order around centroid
    if len(unique_vertices) >= 3:
        centroid = np.mean(unique_vertices, axis=0)
        angles = []
        for vertex in unique_vertices:
            angle = np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
            angles.append(angle)
        
        sorted_indices = np.argsort(angles)
        unique_vertices = [unique_vertices[i] for i in sorted_indices]
    
    return unique_vertices

def create_polytope_boundary_lines(w1, b1, w2=None, b2=None, layer=1, neuron_idx=0, extent=1):
    """
    Create complete boundary lines around each polytope face.
    """
    boundary_lines = []
    
    # Get all regions (polytope faces)
    regions = get_activation_regions(w1, b1, extent)
    
    for region in regions:
        # Find vertices of this region
        vertices = find_region_vertices(region['constraints'], extent)
        
        if len(vertices) >= 3:
            # Create boundary lines around the perimeter of this face
            for i in range(len(vertices)):
                start_vertex = vertices[i]
                end_vertex = vertices[(i + 1) % len(vertices)]  # Wrap around to first vertex
                
                line = Line(
                    start=[start_vertex[0], start_vertex[1], 0],
                    end=[end_vertex[0], end_vertex[1], 0],
                    color=WHITE,
                    stroke_width=2
                )
                boundary_lines.append(line)
    
    # For second layer, also add boundaries where the second layer neuron switches
    if layer == 2 and w2 is not None and b2 is not None:
        for region in regions:
            pattern = region['pattern']
            
            # Calculate effective weights for this region  
            w_eff_x = 0
            w_eff_y = 0
            b_eff = b2[neuron_idx]
            
            for k, active in enumerate(pattern):
                if active:
                    w_eff_x += w2[neuron_idx, k] * w1[k, 0]
                    w_eff_y += w2[neuron_idx, k] * w1[k, 1]
                    b_eff += w2[neuron_idx, k] * b1[k]
            
            # Find where second layer neuron switches within this region
            joint_points = get_relu_joint(w_eff_x, w_eff_y, b_eff, extent)
            if joint_points and len(joint_points) >= 2:
                clipped_points = clip_line_to_region(joint_points, region, extent)
                if clipped_points and len(clipped_points) >= 2:
                    line = Line(
                        start=[clipped_points[0][0], clipped_points[0][1], 0],
                        end=[clipped_points[1][0], clipped_points[1][1], 0],
                        color=RED,
                        stroke_width=3
                    )
                    boundary_lines.append(line)
    
    return boundary_lines

def clip_line_to_region(line_points, region, extent):
    """
    Clip a line segment to a region defined by constraints.
    """
    if not line_points or len(line_points) < 2:
        return []
    
    start, end = line_points[0], line_points[1]
    
    # Sample points along the line and keep those in the region
    valid_points = []
    for t in np.linspace(0, 1, 200):
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Check if point satisfies all region constraints
        valid = True
        for constraint in region['constraints']:
            w1, w2, bias, op = constraint
            value = w1 * x + w2 * y + bias
            if op == '<=' and value > 1e-8:
                valid = False
                break
            elif op == '>=' and value < -1e-8:
                valid = False
                break
        
        if valid and abs(x) <= extent and abs(y) <= extent:
            valid_points.append([x, y])
    
    # Return endpoints of valid segment
    if len(valid_points) >= 2:
        return [valid_points[0], valid_points[-1]]
    else:
        return []

def create_region_polygons(w1, b1, w2=None, b2=None, layer=1, neuron_idx=0, extent=1):
    """
    Create polygon objects for each region of the polytope.
    Useful for shading different regions with different colors/textures.
    """
    regions = get_activation_regions(w1, b1, extent)
    polygons = []
    
    colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
    
    for i, region in enumerate(regions):
        vertices = find_region_vertices(region['constraints'], extent)
        
        if len(vertices) >= 3:
            # Create polygon
            vertices_3d = [[v[0], v[1], 0] for v in vertices]
            polygon = Polygon(*vertices_3d)
            polygon.set_fill(colors[i % len(colors)], opacity=0.3)
            polygon.set_stroke(colors[i % len(colors)], width=2)
            
            polygons.append({
                'polygon': polygon,
                'region': region,
                'vertices': vertices
            })
    
    return polygons

def create_complete_polytope_outlines(w1, b1, w2=None, b2=None, layer=1, neuron_idx=0, extent=1):
    """
    Create complete polygon outlines for each polytope face.
    Returns a list of polygon objects that outline each face.
    """
    regions = get_activation_regions(w1, b1, extent)
    outlines = []
    
    colors = [WHITE, GREY, CHILL_BROWN, FRESH_TAN]
    
    for i, region in enumerate(regions):
        vertices = find_region_vertices(region['constraints'], extent)
        
        if len(vertices) >= 3:
            # Create vertices in 3D
            vertices_3d = [[v[0], v[1], 0] for v in vertices]
            
            # Create polygon outline (no fill, just stroke)
            polygon_outline = Polygon(*vertices_3d)
            polygon_outline.set_fill(opacity=0)  # No fill
            polygon_outline.set_stroke(colors[i % len(colors)], width=3)
            
            outlines.append(polygon_outline)
            
            # Also create individual line segments for more control
            for j in range(len(vertices)):
                start_vertex = vertices[j]
                end_vertex = vertices[(j + 1) % len(vertices)]
                
                line = Line(
                    start=[start_vertex[0], start_vertex[1], 0],
                    end=[end_vertex[0], end_vertex[1], 0],
                    color=WHITE,
                    stroke_width=2
                )
                outlines.append(line)
    
    return outlines





def get_relu_joint(weight_1, weight_2, bias, extent=1):
    if np.abs(weight_2) < 1e-8: 
        x_intercept = -bias / weight_1
        return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
    elif np.abs(weight_1) < 1e-8:
        y_intercept = -bias / weight_2
        return [[-extent, y_intercept], [extent, y_intercept]] if -extent <= y_intercept <= extent else []
    else:
        points = []
        for x in [-extent, extent]:
            y = (-x * weight_1 - bias) / weight_2
            if -extent <= y <= extent: points.append([x, y])
        for y in [-extent, extent]:
            x = (-y * weight_2 - bias) / weight_1
            if -extent <= x <= extent: points.append([x, y])
        unique_points = []
        for p in points:
            is_duplicate = False
            for existing in unique_points:
                if abs(p[0] - existing[0]) < 1e-8 and abs(p[1] - existing[1]) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        return unique_points

def line_from_joint_points_1(joint_points):
    if joint_points:
        # Create 3D points for the joint line
        joint_3d_points = []
        for point in joint_points:
            x, y = point
            z = 0
            joint_3d_points.append([x, y, z])
        
        if len(joint_3d_points) >= 2:
            joint_line = DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=WHITE,
                stroke_width=3,
                dash_length=0.05
            )
            return joint_line


def surface_func_general(u, v, w1, w2, b, viz_scale=0.5):
    linear_output = w1 * u + w2 * v + b
    relu_output = max(0, linear_output)
    z = relu_output * viz_scale 
    return np.array([u, v, z])


def surface_func_second_layer(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
    """
    Surface function for second layer neurons that combines first layer outputs.
    
    Args:
        u, v: Input coordinates (-1 to 1)
        w1: First layer weights (2x2 matrix)
        b1: First layer biases (2 element array)
        w2: Second layer weights (2x2 matrix) 
        b2: Second layer biases (2 element array)
        neuron_idx: Which second layer neuron (0 or 1)
        viz_scale: Scaling factor for visualization
    """
    
    # First layer neuron 1 output
    linear_output_1 = w1[0,0] * u + w1[0,1] * v + b1[0]
    relu_output_1 = max(0, linear_output_1)
    
    # First layer neuron 2 output  
    linear_output_2 = w1[1,0] * u + w1[1,1] * v + b1[1]
    relu_output_2 = max(0, linear_output_2)
    
    # Second layer neuron computation
    second_layer_input = w2[neuron_idx,0] * relu_output_1 + w2[neuron_idx,1] * relu_output_2 + b2[neuron_idx]
    second_layer_output = max(0, second_layer_input)
    
    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return np.array([u, v, z])


def get_second_layer_joints(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Calculate joint lines for second layer neurons.
    
    Returns a list of line segments representing the boundaries where
    the second layer neuron's output changes behavior.
    """
    
    joint_lines = []
    
    # First, get the first layer joint lines (these create discontinuities)
    joint_points_1 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent)
    joint_points_2 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent)
    
    # Add first layer joints as they create surface discontinuities
    if joint_points_1:
        joint_lines.append(joint_points_1)
    if joint_points_2:
        joint_lines.append(joint_points_2)
    
    # Now find where the second layer neuron itself switches on/off
    # This happens when: w2[neuron_idx,0] * relu1 + w2[neuron_idx,1] * relu2 + b2[neuron_idx] = 0
    
    # We need to check this condition in each region defined by the first layer ReLUs
    regions = [
        (False, False),  # Both first layer neurons off
        (True, False),   # Only first neuron on
        (False, True),   # Only second neuron on  
        (True, True)     # Both neurons on
    ]
    
    for relu1_active, relu2_active in regions:
        # Calculate the second layer boundary in this region
        w2_eff = w2[neuron_idx, :]
        b2_eff = b2[neuron_idx]
        
        if relu1_active and relu2_active:
            # Both active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            # Simplifies to: (w2[0]*w1[0,0] + w2[1]*w1[1,0])*u + (w2[0]*w1[0,1] + w2[1]*w1[1,1])*v + (w2[0]*b1[0] + w2[1]*b1[1] + b2) = 0
            eff_w1 = w2_eff[0] * w1[0,0] + w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[0] * w1[0,1] + w2_eff[1] * w1[1,1]
            eff_b = w2_eff[0] * b1[0] + w2_eff[1] * b1[1] + b2_eff
            
        elif relu1_active and not relu2_active:
            # Only first active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + b2 = 0
            eff_w1 = w2_eff[0] * w1[0,0]
            eff_w2 = w2_eff[0] * w1[0,1]
            eff_b = w2_eff[0] * b1[0] + b2_eff
            
        elif not relu1_active and relu2_active:
            # Only second active: w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            eff_w1 = w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[1] * w1[1,1]
            eff_b = w2_eff[1] * b1[1] + b2_eff
            
        else:
            # Neither active: b2 = 0
            if abs(b2_eff) < 1e-8:
                # Always on the boundary (degenerate case)
                continue
            else:
                # Never on the boundary in this region
                continue
        
        # Get joint points for this effective linear function
        region_joints = get_relu_joint(eff_w1, eff_w2, eff_b, extent)
        
        if region_joints:
            # Need to clip these joints to the actual region where this applies
            clipped_joints = clip_joint_to_region(region_joints, relu1_active, relu2_active, w1, b1, extent)
            if clipped_joints:
                joint_lines.append(clipped_joints)
    
    return joint_lines

def clip_joint_to_region(joint_points, relu1_active, relu2_active, w1, b1, extent):
    """
    Clip joint line to the region where the specified ReLU conditions hold.
    """
    if not joint_points or len(joint_points) < 2:
        return []
    
    # Sample points along the joint line
    start, end = joint_points[0], joint_points[1]
    clipped_points = []
    
    # Check multiple points along the line
    for t in np.linspace(0, 1, 100):
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Check if this point satisfies the region conditions
        linear1 = w1[0,0] * x + w1[0,1] * y + b1[0]
        linear2 = w1[1,0] * x + w1[1,1] * y + b1[1]
        
        relu1_here = linear1 > 0
        relu2_here = linear2 > 0
        
        if relu1_here == relu1_active and relu2_here == relu2_active:
            clipped_points.append([x, y])
    
    # Return start and end of the clipped segment
    if len(clipped_points) >= 2:
        return [clipped_points[0], clipped_points[-1]]
    else:
        return []

def create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Create Manim objects for all second layer joint lines.
    """
    joint_lines = get_second_layer_joints(w1, b1, w2, b2, neuron_idx, extent)
    
    manim_lines = []
    colors = [RED, BLUE, GREEN, YELLOW, PURPLE]  # Different colors for different joints
    
    for i, joint_points in enumerate(joint_lines):
        if len(joint_points) >= 2:
            color = colors[i % len(colors)]
            joint_line = DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=color,
                stroke_width=3,
                dash_length=0.05
            )
            manim_lines.append(joint_line)
    
    return manim_lines


class plane_folding_sketch_1(InteractiveScene):
    def construct(self):

        #nice trained 2 hidden layer model. 
        w1=np.array([[-0.02866297,  1.6250265 ],
             [-1.3056537 ,  0.46831134]], dtype=np.float32)
        b1=np.array([-0.4677289,  1.0067637], dtype=np.float32)
        w2=np.array([[ 1.3398709 ,  0.68694556],
                     [-0.29886743, -1.8411286 ]], dtype=np.float32)
        b2=np.array([-0.7817721 ,  0.90856946], dtype=np.float32)
        w3=np.array([[ 1.8897862,  3.0432484],
                     [-1.7220999, -2.2057745]], dtype=np.float32)
        b3=np.array([-1.0249746 ,  0.61326534], dtype=np.float32)
        
        #Not using this flat map right now - but will probably want it
        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.set_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.set_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(ORIGIN)

        self.frame.reorient(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        # self.add(map_img)
        # self.wait()

        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=0.3)
        bent_surface_11 = ParametricSurface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts11=TexturedSurface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts11.set_shading(0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.8)
        group_11=Group(ts11, joint_line_11)

        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=0.3)
        bent_surface_12 = ParametricSurface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts12=TexturedSurface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts12.set_shading(0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.8)
        group_12=Group(ts12, joint_line_12)


        self.add(group_11)
        self.add(group_12)    
        group_12.move_to([0, 0, 1.2])
        self.wait()         

        # Ok scale/flip/add!
        # I almost want liek 2 sets of copies now?
        # One for each combination to come together?
        # Let me start with the first one and then see what's up. 


        surface_func_21 = partial(
            surface_func_second_layer, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=1, viz_scale=0.3
        )


        self.wait()

        bent_surface_21 = ParametricSurface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21 = TexturedSurface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts21.set_shading(0,0,0)
        ts21.set_opacity(0.75)

        joint_lines_21 = create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=1, extent=1)
        group_21_lines = Group(*joint_lines_21)
        group_21=Group(ts21, group_21_lines)

        group_21.move_to([0, 0, 2.4])
        self.add(group_21)  




        # Add complete polygon outlines
        outlines = create_complete_polytope_outlines(w1, b1, w2, b2, 1, 1)
        for outline in outlines:
            outline.shift([0, 0, z_offset])
            self.add(outline)

        # Optionally add filled regions with transparency
        regions = create_region_polygons(w1, b1, w2, b2, 1, 1)
        for region_data in regions:
            polygon = region_data['polygon']
            polygon.shift([0, 0, z_offset])
            polygon.set_fill(opacity=0.1)  # Very transparent
            self.add(polygon)


        self.wait()
        # boundary_lines = create_polytope_boundary_lines(w1, b1, w2, b2, layer=1, neuron_idx=1)
        # boundary_group = Group(*boundary_lines)
        # boundary_group.move_to([0, 0, 2.4])

        # self.wait()

        # Submerging a surface into a very smooth fully opaque liquid...
        # That's not quite right though, becuase values below the surface get clipped to 0
        # Might be interesting/cool to see shaded regions of the map move around and like come together in 
        # different ways


        # Ok great, bending looks good. And i think oreintation is actually mayby right?
        # Contour lines or heatmaps could be nice -> I think for now just a fold line would be a good starting point

        # self.add(bent_surface)
        
        # Move camera to 3D view
        # self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES)
        
        # Animate the transformation from flat to bent
        # self.play(
        #     Transform(map_img, bent_surface),
        #     run_time=3
        # )

        self.wait(20)
        self.embed()











