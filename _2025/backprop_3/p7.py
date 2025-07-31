from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *
from manimlib import *

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


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


def order_closed_loops_with_closure(segments, tol=1e-6):
    """
    Like before, but returns each loop as an (N,3) array of vertices
    where the first point is repeated at the end to explicitly close the loop.
    """
    used = [False] * len(segments)
    loops_pts = []

    for i, seg in enumerate(segments):
        if used[i]:
            continue

        # build ordered, oriented segment list for this loop
        loop_segs = [seg.copy()]
        used[i] = True
        start_pt = seg[0].copy()
        curr_pt = seg[1].copy()

        while True:
            found = False
            for j, seg2 in enumerate(segments):
                if used[j]:
                    continue
                p0, p1 = seg2[0], seg2[1]

                if np.linalg.norm(p0 - curr_pt) < tol:
                    loop_segs.append(seg2.copy())
                    curr_pt = p1.copy()
                    used[j] = True
                    found = True
                    break

                if np.linalg.norm(p1 - curr_pt) < tol:
                    rev = seg2[::-1].copy()
                    loop_segs.append(rev)
                    curr_pt = rev[1].copy()
                    used[j] = True
                    found = True
                    break

            # stop if no continuation or we’re back at start
            if not found or np.linalg.norm(curr_pt - start_pt) < tol:
                break

        # collect vertices and explicitly close the loop
        pts = [loop_segs[0][0]]
        for s in loop_segs:
            pts.append(s[1])

        # if it didn’t naturally close, append the start_pt
        if np.linalg.norm(pts[-1] - start_pt) > tol:
            pts.append(start_pt)

        loops_pts.append(np.vstack(pts))

    return loops_pts




class p7a(InteractiveScene):
    def construct(self):
        model_path='_2025/backprop_3/models/8_2.pth'
        model = BaarleNet([8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.05]
        num_neurons=[8, 8, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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


        #Ok now I need to get my 8 surfaces and ReLu joints for my 8 first layer neurons. 
        first_layer_groups=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.8)


        self.wait() 
        self.frame.reorient(-4, 60, 0, (2.47, 0.04, -0.01), 8.00)
        self.add(first_layer_groups)

        surfaces[2][0].shift([3,0,0.6])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2])
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.601]) #Move slightly above map

        surfaces[2][1].shift([3,0,-0.6])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2])
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, -0.599]) #Move slightly above map

        self.wait()
        # self.play(*[ReplacementTransform(surfaces[1][i].copy(), surfaces[2][0]) for i in range(len(surfaces[1]))],
        #     run_time=3.0)


        surfaces_1_copy=surfaces[1].copy()
        surfaces_1_copy_2=surfaces[1].copy()
        self.add(surfaces_1_copy)
        self.add(surfaces_1_copy_2)

        first_layer_groups_flat=create_first_layer_relu_groups(model, surfaces, num_neurons_first_layer=8, extent=1, vertical_spacing=0.0)
        shifted_line_copies=Group()
        for i in range(len(first_layer_groups_flat)):
            if len(first_layer_groups_flat[i])>1:
                shifted_line_copies.add(first_layer_groups_flat[i][1].copy())
        shifted_line_copies_2 = shifted_line_copies.copy()
        shifted_line_copies.shift([3,0,0.9])

        og_lines=Group()
        for i in range(len(first_layer_groups)):
            # print(len(first_layer_groups[i]))
            if len(first_layer_groups[i])>1:
                og_lines.add(first_layer_groups[i][1])

        og_line_copies=og_lines.copy()
        og_line_copies_2=og_lines.copy()
        self.add(og_line_copies)

        #Start going forward looking
        self.frame.reorient(0, 67, 0, (3.11, 0.28, 0.16), 7.05)
        self.wait()

        #Move overhead to make my animation look less crappy
        self.play(self.frame.animate.reorient(-2, 38, 0, (3.02, 0.27, 0.15), 7.05), run_time=3)


        #ok this is definitely not perfect - but I think I gotta ship it -> it's at minimum shippabale level I would say. 
        self.wait()
        self.play(*[ReplacementTransform(og_line_copies[j], shifted_line_copies[j]) for j in range(len(shifted_line_copies))]+
                    [ReplacementTransform(surfaces_1_copy[i], surfaces[2][0]) for i in range(len(surfaces[1]))],
                    run_time=2) #Play fast becuase it kidna sucks lol
        self.remove(shifted_line_copies); self.add(polygons_21)
        self.wait()


        #I think slight camera move while I do the second animation?
        shifted_line_copies_2.shift([3,0,-0.7])
        self.add(og_line_copies_2)

        self.wait()
        self.play(*[ReplacementTransform(og_line_copies_2[j], shifted_line_copies_2[j]) for j in range(len(shifted_line_copies_2))]+
                    [ReplacementTransform(surfaces_1_copy_2[i], surfaces[2][1]) for i in range(len(surfaces[1]))],
                    self.frame.animate.reorient(-3, 61, 0, (2.87, 0.24, 0.08), 7.01),
                    run_time=2) #Play fast becuase it kidna sucks lol
        self.remove(shifted_line_copies_2); self.add(polygons_22)
        self.wait()


        #Now bring together on same axis - I think with copies this time, so I can just swap stuff out as we get bigger
        #Drop opacity on input maps and lines too!

        # Hmm hmm hmm maybe I dont' need a third panel -> maybe it's enough to bring them together like I did last time??
        # Then I'm just showing final surfaces and border as we expand -> that might be kinda a nice vibe actually. 


        # So it's definitely not that easy to see the border here -> i should briefly look at my top polytope option I think!
        # We could have them merge into this, or cross fade to it or something -> let me mess around here. 


        self.wait()


        top_polygons_vgroup=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        self.wait()

        self.play(surfaces[2][0].animate.shift([0, 0, -0.6]), 
                  surfaces[2][1].animate.shift([0, 0, 0.6]), 
                  polygons_21.animate.shift([0, 0, -0.6]).set_color(BLUE),
                  polygons_22.animate.shift([0, 0, 0.6]).set_color(YELLOW),
                  surfaces[1].animate.set_opacity(0.1), 
                  og_lines.animate.set_opacity(0.2),
                  self.frame.animate.reorient(1, 42, 0, (3.0, 0.05, -0.17), 3.54),
                  run_time=3)
        self.add(top_polygons_vgroup) #This actually worked pretty well!


        loops=order_closed_loops_with_closure(intersection_lines)
        lines=VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])

        self.wait()
        self.play(ShowCreation(lines), run_time=3)
        self.wait()


        ## Ok ok ok getting close ot finishing the 8 neuron model -> it's messy but I do think 
        ## that this will be a cool opening. 
        ## Last move I think is: 
        ## Move cameara to the right as I make the copy
        ## Toally fade out layer 1, make a copy of the map and final borders, flatten both and move a copy over
        ## Then as I step through bigger models I'll just render these 2 things!
        ## Then on P8 I can move to nice overhead view or something fun on the 512 map. 


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        flat_map.set_shading(0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])

        self.wait()
        self.play(ReplacementTransform(lines.copy(), lines_flat), 
                 ReplacementTransform(surfaces[2][0], flat_map), 
                  surfaces[1].animate.set_opacity(0.0), 
                  og_lines.animate.set_opacity(0.0),
                 self.frame.animate.reorient(0, 45, 0, (4.28, 0.08, -0.19), 3.97), 
                 run_time=3)
        self.add(lines_flat) #Occlusions
        self.wait()

        self.wait(20)
        self.embed()

        #Ok sweet -> now i just need these same static images for the bigger models. 
        # A few tweaks I might want to make, but I think this is good for now. 




class p7b(InteractiveScene):
    def construct(self):
        model_path='_2025/backprop_3/models/16_2.pth'
        model = BaarleNet([16])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.07, 0.07, 0.04]
        num_neurons=[16, 16, 2]

        #Precompute my surfaces, and polygons moving through network
        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
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

        


        #I guess I don't have to do this every time -> coudl just draw all the lines? Might be better when we get really big here
        loops=order_closed_loops_with_closure(intersection_lines)
        lines=VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([3, 0, 0])


        top_polygons_vgroup=VGroup()
        for j, p in enumerate(my_top_polygons):
            if len(p)<3: continue
            if my_indicator[j]: color=YELLOW
            else: color=BLUE
            
            p_scaled=copy.deepcopy(p) #Scaling for viz
            p_scaled[:,2]=p_scaled[:,2]*viz_scales[2]
            poly_3d = Polygon(*p_scaled,
                             fill_color=color,
                             fill_opacity=0.4,
                             stroke_color=color,
                             stroke_width=2)
            poly_3d.set_opacity(0.3)
            poly_3d.shift([3, 0, 0])
            top_polygons_vgroup.add(poly_3d)

        surfaces[2][0].shift([3,0,0])
        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2])
        polygons_21_copy=polygons_21.copy()
        polygons_21.shift([3, 0, 0.001]) #Move slightly above map

        surfaces[2][1].shift([3,0,0])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2])
        polygons_22_copy=polygons_22.copy()
        polygons_22.shift([3, 0, 0.002]) #Move slightly above map

        polygons_22.set_color(YELLOW)
        polygons_21.set_color(BLUE)

        polygons_21.set_opacity(0.1)
        polygons_22.set_opacity(0.1)
        surfaces[2][0].set_opacity(0.4)
        surfaces[2][1].set_opacity(0.4)
        top_polygons_vgroup.set_opacity(0.5)

        self.frame.reorient(0, 45, 0, (4.28, 0.08, -0.19), 3.97)
        self.add(surfaces[2][0], surfaces[2][1])
        self.add(polygons_21, polygons_22)
        self.add(top_polygons_vgroup)
        self.add(lines)


        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        flat_map.set_shading(0,0,0).set_opacity(0.8)
        flat_map.shift([5.7, 0, 0])

        lines_flat=VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, 0])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines_flat.add(line)
        lines_flat.shift([5.7, 0, 0])


        self.add(flat_map, lines_flat)
        self.wait()



        self.wait(20)
        self.embed()




















        # self.add(top_polygons_vgroup)


        # self.play(ReplacementTransform(polygons_21, top_polygons_vgroup), run_time=3


        # polygons_22.set_opacity(0.4)



        # self.play(polygons_21.animate.set_color(BLUE), run_time=1.5)
        # self.play(polygons_22.animate.set_color(YELLOW), run_time=1.5)

        # self.remove(netherlands_label, belgium_label)
        # self.play(polygons_21.animate.shift([0, 0, -0.8]), 
        #           surfaces[2][0].animate.shift([0, 0, -0.8]), 
        #           axes_1.animate.shift([0, 0, -0.8]), 
        #           polygons_22.animate.shift([0, 0, 0.8]),
        #           surfaces[2][1].animate.shift([0, 0, 0.8]), 
        #           axes_2.animate.shift([0, 0, 0.8]),
        #           # self.frame.animate.reorient(-7, 40, 0, (3.21, 0.51, -0.66), 3.95), 
        #           group_11[0].animate.set_opacity(0.0),
        #           group_11[1].animate.set_opacity(0.0),
        #           group_12[0].animate.set_opacity(0.0),
        #           group_12[1].animate.set_opacity(0.0),
        #           group_13[0].animate.set_opacity(0.0),
        #           group_13[1].animate.set_opacity(0.0),      
        #           run_time=3)
        # self.wait()





        # polygons_21_copy.shift([6, 0, 0])
        # polygons_22_copy.shift([6, 0, 0])
        # polygons_21_copy.set_color(BLUE)
        # polygons_22_copy.set_color(YELLOW)

        # loops=order_closed_loops_with_closure(intersection_lines)

        # lines=VGroup()
        # for loop in loops: 
        #     loop=loop*np.array([1, 1, viz_scales[2]])
        #     line = VMobject()
        #     line.set_points_as_corners(loop)
        #     line.set_stroke(color='#FF00FF', width=5)
        #     lines.add(line)
        # lines.shift([6, 0, 0])

        # self.wait()
        # self.play(ReplacementTransform(polygons_21.copy(), polygons_21_copy), 
        #           ReplacementTransform(polygons_22.copy(), polygons_22_copy), 
        #           surfaces[2][0].copy().animate.move_to([6,0,0]), 
        #           surfaces[1].animate.set_opacity(0.1), 
        #           og_lines.animate.set_opacity(0.2),
        #           self.frame.animate.reorient(-2, 57, 0, (4.84, -0.09, 0.01), 5.02),
        #           run_time=3)
        # self.play(ShowCreation(lines), run_time=3)
        # self.wait()







        # intersection_points_raveled=np.array(intersection_lines).reshape(-1, 3)

        # intersection_points_raveled=intersection_points_raveled*np.array([1, 1, viz_scales[2]])
        # intersection_points_raveled=intersection_points_raveled[(2, 6, 5, 1, 0),:] #Change ordering for smooth animation in
        # self.add(polygons_21_copy, polygons_22_copy)
        # self.play(ShowCreation(lines))





        # self.play(*[ReplacementTransform(surfaces_1_copy[i], surfaces[2][0]) for i in range(len(surfaces[1]))]+
        #            [Transform(og_line_copies[j], shifted_line_copies[j]) for j in range(len(shifted_line_copies))],
        #            # [og_line_copies[j].animate.move_to(shifted_line_copies[j]) for j in range(len(shifted_line_copies))],
        #     run_time=3.0)




        # og_line_copies.shift([1,1,1])
        # og_line_copies.set_opacity(1.0)

        # self.play(*[og_line_copies[j].animate.move_to(shifted_line_copies[j]) for j in range(len(shifted_line_copies))])





        # self.add(shifted_line_copies)



        # shifted_line_copies=Group(*[first_layer_groups[i][1].copy() for i in range(len(first_layer_groups))])

        # for i in range(8): print(len(first_layer_groups[i]))

        # # Create the mapped lines for the first output neuron
        # mapped_relu_lines = map_relu_lines_to_surface(first_layer_groups, surface_funcs[2][0])
        # mapped_relu_lines.shift([3, 0, 0.6])  # Same shift as surfaces[2][0]


        # self.add(mapped_relu_lines)



        # self.play(ReplacementTransform(first_layer_groups[0][1].copy(), mapped_relu_lines[0]))


        # # Then animate the transformation
        # self.play(
        #     *[ReplacementTransform(surfaces[1][i].copy(), surfaces[2][0]) for i in range(len(surfaces[1]))],
        #     *[ReplacementTransform(first_layer_groups[i][1].copy(), mapped_relu_lines[i]) 
        #       for i in range(len(first_layer_groups)) if len(first_layer_groups[i]) > 1],
        #     run_time=3.0
        # )





        # mapped_relu_lines = create_mapped_relu_lines(first_layer_groups, surfaces, model, target_neuron_idx=0)
        # mapped_relu_lines.shift([3, 0, 0.6])  # Same shift as surfaces[2][0]

        # self.add(mapped_relu_lines)

        # self.play(
        #     *[ReplacementTransform(surfaces[1][i].copy(), surfaces[2][0]) for i in range(len(surfaces[1]))],
        #     *[ReplacementTransform(first_layer_groups[i][1].copy(), mapped_relu_lines[i]) 
        #       for i in range(len(first_layer_groups)) if len(first_layer_groups[i]) > 1],
        #     run_time=3.0
        # )


        # self.wait()
        # self.play(*[ReplacementTransform(surfaces[1][i].copy(), surfaces[2][0]) for i in range(len(surfaces[1]))],
        #     run_time=3.0)



        # self.add(surfaces[2][0], polygons_21)        
        # self.add(surfaces[2][1], polygons_22)


        # self.wait()

        # Hmm how am I going to animate these freaking Relu lines over...I think they can turn into polygons once they get there
        # but they gotta move!










