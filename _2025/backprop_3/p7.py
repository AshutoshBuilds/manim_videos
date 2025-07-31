from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

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
        polygons_21.shift([3, 0, 0.601]) #Move slightly above map

        surfaces[2][1].shift([3,0,-0.6])
        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2])
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
        shifted_line_copies_2.shift([3,0,-0.6])
        self.add(og_line_copies_2)

        self.wait()
        self.play(*[ReplacementTransform(og_line_copies_2[j], shifted_line_copies_2[j]) for j in range(len(shifted_line_copies_2))]+
                    [ReplacementTransform(surfaces_1_copy_2[i], surfaces[2][1]) for i in range(len(surfaces[1]))],
                    self.frame.animate.reorient(-1, 56, 0, (3.02, 0.27, 0.15), 7.05),
                    run_time=2) #Play fast becuase it kidna sucks lol
        self.remove(shifted_line_copies_2); self.add(polygons_22)
        self.wait()



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








        self.wait(20)
        self.embed()


