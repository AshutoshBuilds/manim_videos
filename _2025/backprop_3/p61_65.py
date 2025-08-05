from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes
from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
colors = [BLUE, GREY, GREEN, TEAL, PURPLE, ORANGE, PINK, TEAL, RED, YELLOW ]
# colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

def create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.9):
    """
    Creates ReLU joint groups for all neurons in the first layer.
    
    Args:
        model: PyTorch model containing the neural network
        surfaces: List of surface groups (surfaces[0] contains first layer surfaces)
        num_neurons_first_layer: Number of neurons in first layer (default 8)
        extent: Extent parameter for ReLU joint calculation (default 1)
        vertical_spacing: Vertical spacing between groups (default 0.3)
    
    Returns:
        Group: A manim Group containing num_neurons_first_layer groups, 
               each with a surface and its corresponding joint line
    """
    
    # Extract weights and biases from first layer (layer 0 in model.model)
    with torch.no_grad():
        w1 = model.model[0].weight.cpu().numpy()  # Shape: [num_neurons, 2]
        b1 = model.model[0].bias.cpu().numpy()    # Shape: [num_neurons]
    
    # Create the main group to hold all neuron groups
    all_relu_groups = Group()
    
    # Calculate total height needed for centering
    total_height = (num_neurons_first_layer - 1) * vertical_spacing
    start_z = total_height / 2  # Start from top
    
    for neuron_idx in range(num_neurons_first_layer):
        # Get ReLU joint points for this neuron
        joint_points = get_relu_joint(w1[neuron_idx, 0], w1[neuron_idx, 1], b1[neuron_idx], extent=extent)
        
        # Create joint line from points
        joint_line = line_from_joint_points_1(joint_points)
        if joint_line is not None: 
            joint_line.set_opacity(0.9)
            neuron_group = Group(surfaces[1][neuron_idx], joint_line)
        else:
            neuron_group = Group(surfaces[1][neuron_idx])
        
        # Position the group vertically
        z_position = start_z - neuron_idx * vertical_spacing
        neuron_group.shift([0, 0, z_position])
        
        # Add to main group
        all_relu_groups.add(neuron_group)
    
    return all_relu_groups


class p61(InteractiveScene):
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_1.pth'
        model = BaarleNet([8,8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=2*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
                ts.set_shading(0,0,0).set_opacity(0.8)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

            #Split polygons w/ Relu and clip negative values to z=0
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            #Merge zero regions
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])

        #Outputs surfaces
        groups_output=Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([6, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

        top_polygons_vgroup=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            poly_3d.shift([9, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines.add(line)
        lines.shift([9, 0, 0.])

        top_polygons_vgroup_flat=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            poly_3d.shift([9, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        flat_map_2.set_shading(0,0,0).set_opacity(0.8)
        flat_map_2.shift([9, 0, -2])

        lines_flat=VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([9, 0, -2])    

        # surface_51.set_opacity(0.2)
        # surface_52.set_opacity(0.9)
        # polygons_51.set_opacity(0.4)
        # polygons_52.set_opacity(0.5)



        # polygons_41=manim_polygons_from_np_list(polygons['2.linear_out'][0], colors=colors_5, viz_scale=viz_scales[4], opacity=0.6)
        # polygons_41.shift([6, 0, 1.501]) #Move slightly above map

        ##Feeling like a fly around of this static scene would be non-terrible?

        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.frame.reorient(0, 64, 0, (4.62, 2.85, -1.51), 12.99)

        self.add(groups_1)
        self.add(layer_1_polygons_flat)
        self.add(groups_2)
        self.add(layer_2_polygons_flat)
        self.add(groups_output)
        self.add(group_combined_output)
        self.add(top_polygons_vgroup)
        self.add(lines)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)

        self.wait()

        #19, 102

        #Focus on tiling
        self.play(self.frame.animate.reorient(0, 58, 0, (1.45, 1.08, -5.07), 5.78), run_time=6)
        self.wait()

        self.play(self.frame.animate.reorient(18, 58, 0, (2.47, 1.79, 0.9), 5.81), 
                  groups_1.animate.set_opacity(0.1), 
                  groups_2[0].animate.set_opacity(0.1),
                  groups_2[2:].animate.set_opacity(0.1),
                  groups_output.animate.set_opacity(0.1), 
                  run_time=6
                  )
        self.wait()

        self.play(self.frame.animate.reorient(-4, 58, 0, (7.86, 1.54, -1.02), 6.99), 
                  groups_1.animate.set_opacity(0.6), 
                  groups_2[0].animate.set_opacity(0.6),
                  groups_2[2:].animate.set_opacity(0.6),
                  groups_output.animate.set_opacity(0.6), 
                  run_time=6
                  )
        self.wait()

        self.play(self.frame.animate.reorient(0, 38, 0, (9.09, 0.29, -2.69), 3.98), 
                 group_combined_output.animate.set_opacity(0.05),
                 top_polygons_vgroup.animate.set_opacity(0.05),
                 lines.animate.set_opacity(0.05),
                 run_time=4)
        self.wait()

        #Overall view
        self.play(self.frame.animate.reorient(0, 74, 0, (5.17, 1.54, -0.87), 11.81), 
                 group_combined_output.animate.set_opacity(0.2),
                 top_polygons_vgroup.animate.set_opacity(0.5),
                 lines.animate.set_opacity(0.8),
                 run_time=6)
        self.wait()




        self.wait(20)
        self.embed()




class p62(InteractiveScene):
    def construct(self):

        model_path='_2025/backprop_3/models/8_8_8_1.pth'
        model = BaarleNet([8,8, 8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[8, 8, 8, 8, 8, 8, 2]
        vertical_spacing=1.0

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=1.0, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=2*min(adaptive_viz_scales[-1]) #little manual ramp here
        adaptive_viz_scales[-1]=[final_layer_viz, final_layer_viz]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx]) #viz_scales[layer_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
                ts.set_shading(0,0,0).set_opacity(0.8)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        #Move polygons through network
        polygons={} #dict of all polygones as we go. 
        polygons['-1.new_tiling']=[np.array([[-1., -1, 0], #First polygon is just input plane
                                            [-1, 1, 0], 
                                            [1, 1, 0], 
                                            [1, -1, 0]])]

        for layer_id in range(len(model.model)//2): #Move polygont through layers     
            polygons[str(layer_id)+'.linear_out']=process_with_layers(model.model[:2*layer_id+1], polygons[str(layer_id-1)+'.new_tiling']) 

            #Split polygons w/ Relu and clip negative values to z=0
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            #Merge zero regions
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            #Compute new tiling
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        groups_1=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=vertical_spacing)


        #Ok let's get everything (including shadows) up first, and then decide if I want to animate the creation of anything?

        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -5.0])

        #Create layer 2 polygons -> I guess post cut right?
        groups_2=Group()
        layer_idx=3
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([3*(layer_idx-1)/2, 0, start_z - neuron_idx * vertical_spacing])
            groups_2.add(g)


        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -5.0])

        #Outputs surfaces
        groups_output=Group()
        layer_idx=len(model.model)-1
        total_height = (num_neurons[layer_idx] - 1) * vertical_spacing
        start_z = total_height / 2  # Start from top
        for neuron_idx in range(num_neurons[layer_idx]):
            # split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_merged'][neuron_idx] for item in sublist]
            pgs=manim_polygons_from_np_list(polygons['2.linear_out'][neuron_idx], colors=colors, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx], opacity=0.6)
            s=surfaces[layer_idx][neuron_idx]
            g=Group(s, pgs[1:]) #Crazy to leave off first merged/flat group here?
            g.shift([6, 0, start_z - neuron_idx * vertical_spacing])
            groups_output.add(g)

        #Output surfaces together
        group_combined_output=groups_output.copy()
        group_combined_output[0].set_color(BLUE)
        group_combined_output[1].set_color(YELLOW)
        group_combined_output[0].shift([3, 0, -vertical_spacing/2])
        group_combined_output[1].shift([3, 0, vertical_spacing/2])

        top_polygons_vgroup=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*adaptive_viz_scales[-1][0] #Flatten that shit!
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            poly_3d.shift([9, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        # loops=order_closed_loops_with_closure(intersection_lines)
        lines=VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=4)
            lines.add(line)
        lines.shift([9, 0, 0.])

        top_polygons_vgroup_flat=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=0 #p_scaled[:,2]*viz_scales[2] #Flatten that shit!
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.5)
            poly_3d.shift([9, 0, -2])
            top_polygons_vgroup_flat.add(poly_3d)

        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map_2=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        flat_map_2.set_shading(0,0,0).set_opacity(0.8)
        flat_map_2.shift([9, 0, -2])

        lines_flat=VGroup()
        for loop in intersection_lines: 
            # loop=np.hstack((loop, np.zeros((len(loop), 1))))
            loop[:,2]=0
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([9, 0, -2])    

        # surface_51.set_opacity(0.2)
        # surface_52.set_opacity(0.9)
        # polygons_51.set_opacity(0.4)
        # polygons_52.set_opacity(0.5)



        # polygons_41=manim_polygons_from_np_list(polygons['2.linear_out'][0], colors=colors_5, viz_scale=viz_scales[4], opacity=0.6)
        # polygons_41.shift([6, 0, 1.501]) #Move slightly above map

        ##Feeling like a fly around of this static scene would be non-terrible?

        group_combined_output.set_opacity(0.3)
        # top_polygons_vgroup.set_opacity(0.6)

        self.wait()
        self.frame.reorient(0, 64, 0, (4.62, 2.85, -1.51), 12.99)

        self.add(groups_1)
        self.add(layer_1_polygons_flat)
        self.add(groups_2)
        self.add(layer_2_polygons_flat)
        self.add(groups_output)
        self.add(group_combined_output)
        self.add(top_polygons_vgroup)
        self.add(lines)
        self.add(flat_map_2)
        self.add(top_polygons_vgroup_flat)
        self.add(lines_flat)

        self.wait()

        #19, 102

        #Focus on tiling
        self.play(self.frame.animate.reorient(0, 58, 0, (1.45, 1.08, -5.07), 5.78), run_time=6)
        self.wait()

        self.play(self.frame.animate.reorient(18, 58, 0, (2.47, 1.79, 0.9), 5.81), 
                  groups_1.animate.set_opacity(0.1), 
                  groups_2[0].animate.set_opacity(0.1),
                  groups_2[2:].animate.set_opacity(0.1),
                  groups_output.animate.set_opacity(0.1), 
                  run_time=6
                  )
        self.wait()

        self.play(self.frame.animate.reorient(-4, 58, 0, (7.86, 1.54, -1.02), 6.99), 
                  groups_1.animate.set_opacity(0.6), 
                  groups_2[0].animate.set_opacity(0.6),
                  groups_2[2:].animate.set_opacity(0.6),
                  groups_output.animate.set_opacity(0.6), 
                  run_time=6
                  )
        self.wait()

        self.play(self.frame.animate.reorient(0, 38, 0, (9.09, 0.29, -2.69), 3.98), 
                 group_combined_output.animate.set_opacity(0.05),
                 top_polygons_vgroup.animate.set_opacity(0.05),
                 lines.animate.set_opacity(0.05),
                 run_time=4)
        self.wait()

        #Overall view
        self.play(self.frame.animate.reorient(0, 74, 0, (5.17, 1.54, -0.87), 11.81), 
                 group_combined_output.animate.set_opacity(0.2),
                 top_polygons_vgroup.animate.set_opacity(0.5),
                 lines.animate.set_opacity(0.8),
                 run_time=6)
        self.wait()




        self.wait(20)
        self.embed()








