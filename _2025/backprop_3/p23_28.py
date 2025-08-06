from functools import partial
import sys

sys.path.append('_2025/backprop_3')
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes


from manimlib import *
# from MF_Tools import *
import glob
import torch


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

svg_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/to_manim'
graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

class p23(InteractiveScene):
    def construct(self):
        '''Ok so I think 23 is very much an extention of 21, so once we come back from overhead table
           the network will still be in the center (but now with ReLu drawn on it - and I'll animate foling up the h(1)
           planes, then I thhink the network goes down or to the corner, and I being back the map...)
        '''
        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()


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


        #Get first layer Relu Joints
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
        group_11=Group(surfaces[1][0], joint_line_11)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
        group_12=Group(surfaces[1][1], joint_line_12)


        # Ok, so we want to start out with room still in the center for the ball and stick diagram
        # With planes not bent yet, no fold lines yet, probably add thos right before folding
        # And coloring/shading to match neuron colors would be cool! Migth be able to just use polygons I've already compute?
        # Hmm also might want axes? Yeah this whole scene kidna feels like we might want axes...
        # Should be ok/fine. Just need to figure out how to map various ish to axes


        axes_1 = ThreeDAxes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=2, height=2, depth=1.5,
            axis_config={"color": FRESH_TAN, "include_ticks": False, "include_numbers": False, "include_tip": True,
                "stroke_width":4, "tip_config": {"width":0.08, "length":0.08}}
                )
        axes_2=axes_1.copy()
        axes_1.move_to([0, 0, 1.7])
        axes_2.move_to([0, 0, -1.7])
        self.frame.reorient(41, 69, 0, (0.04, -0.02, 0.26), 5.07) #Ok this FoV seems not terrible?
        self.add(axes_1, axes_2)

        

        # axes_1.c2p?
        polygons_11_pts=[]
        for p in polygons['0.linear_out'][0][0]:
            p[2]=p[2]*viz_scales[0]
            polygons_11_pts.append(axes_1.c2p(*p))
        polygons_11_pts=np.array(polygons_11_pts)
        polygons_11=manim_polygons_from_np_list([polygons_11_pts], colors=[CYAN], viz_scale=1, opacity=0.6)
        self.add(polygons_11)





        group_11.move_to([0, 0, 1.5])
        group_12.move_to([0, 0, -1.5])



        self.add(group_11)
        self.add(group_12)


        self.wait()







        group_11.shift([0, 0, 1.5])






        self.wait()







        self.wait(20)
        self.embed()
















