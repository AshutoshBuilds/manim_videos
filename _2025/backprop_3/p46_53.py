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
# colors = [BLUE, GREY, GREEN, TEAL, PURPLE, ORANGE, PINK, TEAL, RED, YELLOW ]
colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

class p46(InteractiveScene):
    def construct(self):

        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        map_img.set_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.set_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(ORIGIN)

        model = BaarleNet([2])

        w1 = np.array([[2.5135, -1.02481],
         [-1.4043, 2.41291]], dtype=np.float32)
        b1 = np.array([-1.23981, -0.450078], dtype=np.float32)
        w2 = np.array([[3.17024, 1.32567],
         [-3.40372, -1.53878]], dtype=np.float32)
        b2 = np.array([-0.884835, 0.0332228], dtype=np.float32)

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))

        viz_scales=[0.2, 0.2, 0.13]
        num_neurons=[2, 2, 2]

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



        #Get first layer Relu Joints - I havea method that automates this if I need ot scale it. 

        # polygons_11=manim_polygons_from_np_list([item for sublist in polygons['0.split_polygons_nested_clipped'][0] for item in sublist], 
        #                                         colors=[GREY, BLACK], viz_scale=viz_scales[1], opacity=0.3)
        # polygons_11.shift([0, 0, 0.001]) #Move slightly above map

        # polygons_12=manim_polygons_from_np_list([item for sublist in polygons['0.split_polygons_nested_clipped'][1] for item in sublist], 
        #                                         colors=[GREY, BLACK], viz_scale=viz_scales[1], opacity=0.3)
        # polygons_12.shift([0, 0, 0.001]) #Move slightly above map


        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=Group(surfaces[1][0]) #, polygons_11)
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=Group(surfaces[1][1]) #, polygons_12)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_11.shift([0, 0, 1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=Group(surfaces[2][0], polygons_21)
        group_22=Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 1.5])
        group_22.shift([3.0, 0, 0])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.75])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.75])

        loops=order_closed_loops_with_closure(intersection_lines)
        lines=VGroup()
        for loop in loops: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([6, 0, 0.75])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)
        group_31[1].set_opacity(0.4)
        group_32[1].set_opacity(0.4)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        self.frame.reorient(-1, 42, 0, (3.1, 0.59, -0.39), 6.92)

        self.wait()
        self.play(FadeIn(group_11), FadeIn(group_12))
        self.wait()

        self.play(FadeIn(group_21), FadeIn(group_22))
        self.wait()

        #Ok now a little animation bringin the two bent surfaces together and changing their colors? 
        self.play(ReplacementTransform(group_21.copy(), group_31), 
                  ReplacementTransform(group_22.copy(), group_32), 
                 run_time=3.0)
        self.play(ShowCreation(lines))
        self.wait()

        #Tempted to zoom in on decision boundary, but mabye we've seen it enough?
        # Ok now need ot hink through how this fits with illustrator network and then exapnding it! Good start!





        # self.add(group_31, group_32, lines)
        # self.wait()

        # self.play(ReplacementTransform(polygons_11[0].copy(), polygons_21[2:]))




        self.wait(20)
        self.embed()