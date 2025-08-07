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


def surface_func_from_model_with_axes(u, v, model, layer_idx, neuron_idx, axes=None, viz_scale=0.5):
    """
    Create a surface function for visualizing activations at any layer.
    
    Args:
        u, v: Input coordinates (in parameter space, typically -1 to 1)
        model: BaarleNet model
        layer_idx: Direct index into model.model (e.g., 0, 1, 2, 3...)
        neuron_idx: Which neuron in that layer
        axes: Manim axes object to use for coordinate transformation
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
        
        # If axes provided, transform to axes coordinate system
        if axes is not None:
            return axes.c2p(u, v, z)
        else:
            return np.array([u, v, z])

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
        axes_2 = ThreeDAxes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=2, height=2, depth=1.5,
            axis_config={"color": FRESH_TAN, "include_ticks": False, "include_numbers": False, "include_tip": True,
                "stroke_width":4, "tip_config": {"width":0.08, "length":0.08}}
                )
        axes_1.move_to([0, 0, 1.7])
        axes_2.move_to([0, 0, -1.7])


        # Ok this is kinda annoying -> i think for this to work with axes like this if have to 
        # do my viz scale before shifting? I guess that makes sense. 
        polygons_11_pts=[]
        for p in polygons['0.linear_out'][0][0]:
            p[2]=p[2]*viz_scales[0]
            polygons_11_pts.append(axes_1.c2p(*p))
        polygons_11_pts=np.array(polygons_11_pts)
        polygons_11=manim_polygons_from_np_list([polygons_11_pts], colors=[CYAN], viz_scale=1, opacity=0.3)

        polygons_12_pts=[]
        for p in polygons['0.linear_out'][1][0]:
            p[2]=p[2]*viz_scales[0]
            polygons_12_pts.append(axes_2.c2p(*p))
        polygons_12_pts=np.array(polygons_12_pts)
        polygons_12=manim_polygons_from_np_list([polygons_12_pts], colors=[YELLOW], viz_scale=1, opacity=0.3)


        #Ok now how do I get my surface and ReLu join onto the same axis?
        # array([[-1.        , -1.        ,  1.27255271],                                                                                                                     
        #        [-1.        ,  0.91999996,  0.98150721],                                                                                                                     
        #        [ 0.91999996,  0.91999996,  1.69534064],                                                                                                                     
        #        [ 0.91999996, -1.        ,  1.98638611]])   

        # In [2]: axes_1.c2p(-1, -1, 0)                                                                                                                                       
        # Out[2]: array([-1.        , -1.        ,  1.65999994])   
        # In [5]: axes_1.c2p(1, 1, 0)                                                                                                                                         
        # Out[5]: array([0.91999996, 0.91999996, 1.65999994]) 

        surface_func_11=partial(surface_func_from_model_with_axes, model=model, layer_idx=0, neuron_idx=0, axes=axes_1, viz_scale=viz_scales[0])
        bent_surface_11 = ParametricSurface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_11=TexturedSurface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        surface_11.set_shading(0,0,0).set_opacity(0.8)

        surface_func_12=partial(surface_func_from_model_with_axes, model=model, layer_idx=0, neuron_idx=1, axes=axes_2, viz_scale=viz_scales[0])
        bent_surface_12 = ParametricSurface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_12=TexturedSurface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        surface_12.set_shading(0,0,0).set_opacity(0.8)

        #Get first layer Relu Joints
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_points_11a=axes_1.c2p(joint_points_11[0][0], joint_points_11[0][1], 0)
        joint_points_11b=axes_1.c2p(joint_points_11[1][0], joint_points_11[1][1], 0)
        joint_line_11 = DashedLine( start=joint_points_11a, end=joint_points_11b, color=WHITE, stroke_width=5, dash_length=0.08)

        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_points_12a=axes_2.c2p(joint_points_12[0][0], joint_points_12[0][1], 0)
        joint_points_12b=axes_2.c2p(joint_points_12[1][0], joint_points_12[1][1], 0)
        joint_line_12 = DashedLine( start=joint_points_12a, end=joint_points_12b, color=WHITE, stroke_width=5, dash_length=0.08)


        #I think we need polygons both pre and post bend!
        polygons_21a=VGroup()
        for i in range(2):
            polygons_21_pts=[]
            for p in polygons['0.split_polygons_nested'][0][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_21_pts.append(axes_1.c2p(*p))
            polygons_21_pts=np.array(polygons_21_pts)
            polygons_21a.add(manim_polygons_from_np_list([polygons_21_pts], colors=[CYAN], viz_scale=1, opacity=0.3))

        polygons_22a=VGroup()
        for i in range(2):
            polygons_22_pts=[]
            for p in polygons['0.split_polygons_nested'][1][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_22_pts.append(axes_2.c2p(*p))
            polygons_22_pts=np.array(polygons_22_pts)
            # print(polygons_22_pts)
            polygons_22a.add(manim_polygons_from_np_list([polygons_22_pts], colors=[YELLOW], viz_scale=1, opacity=0.3))

        polygons_21b=VGroup()
        for i in range(2):
            polygons_21_pts=[]
            for p in polygons['0.split_polygons_nested_clipped'][0][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_21_pts.append(axes_1.c2p(*p))
            polygons_21_pts=np.array(polygons_21_pts)
            polygons_21b.add(manim_polygons_from_np_list([polygons_21_pts], colors=[CYAN], viz_scale=1, opacity=0.3))

        polygons_22b=VGroup()
        for i in range(2):
            polygons_22_pts=[]
            for p in polygons['0.split_polygons_nested_clipped'][1][0][i]:
                p[2]=p[2]*viz_scales[0]
                polygons_22_pts.append(axes_2.c2p(*p))
            polygons_22_pts=np.array(polygons_22_pts)
            # print(polygons_22_pts)
            polygons_22b.add(manim_polygons_from_np_list([polygons_22_pts], colors=[YELLOW], viz_scale=1, opacity=0.3))


        surface_func_21=partial(surface_func_from_model_with_axes, model=model, layer_idx=1, neuron_idx=0, axes=axes_1, viz_scale=viz_scales[0])
        bent_surface_21 = ParametricSurface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_21=TexturedSurface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        surface_21.set_shading(0,0,0).set_opacity(0.8)

        surface_func_22=partial(surface_func_from_model_with_axes, model=model, layer_idx=1, neuron_idx=1, axes=axes_2, viz_scale=viz_scales[0])
        bent_surface_22 = ParametricSurface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        surface_22=TexturedSurface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        surface_22.set_shading(0,0,0).set_opacity(0.8)

        #Need these for a move later
        surface_11_copy=surface_11.copy()
        polygons_21a_copy=polygons_21a.copy()
        polygons_11_copy=polygons_11.copy()
        surface_12_copy=surface_12.copy()
        polygons_22a_copy=polygons_22a.copy()
        polygons_12_copy=polygons_12.copy()
        surface_21_copy=surface_21.copy()
        surface_22_copy=surface_22.copy()


        self.frame.reorient(-53, 68, 0, (0.01, -0.1, 0.09), 5.58)

        self.add(surface_11, polygons_11, axes_1) #, joint_line_11)
        self.add(surface_12, polygons_12, axes_2) #, joint_line_12)

        self.wait()
        self.remove(polygons_11); self.add(polygons_21a); self.add(joint_line_11)
        self.play(ReplacementTransform(polygons_21a[1], polygons_21b[1]), 
                  ReplacementTransform(surface_11, surface_21), 
                  # FadeIn(joint_line_11),
                  run_time=3.2)
        self.remove(polygons_21b); self.remove(polygons_21a); self.add(polygons_21b)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(axes_1); self.add(axes_1)
        self.wait()

        self.remove(polygons_12); self.add(polygons_22a); self.add(joint_line_12)
        self.play(ReplacementTransform(polygons_22a[1], polygons_22b[1]), 
                  ReplacementTransform(surface_12, surface_22), 
                  # FadeIn(joint_line_12),
                  run_time=3.2)
        self.remove(polygons_22b); self.remove(polygons_22a); self.add(polygons_22b)
        self.remove(joint_line_12); self.add(joint_line_12)
        self.remove(axes_2); self.add(axes_2)
        self.wait()

        #Ok dope dope dope - now we move thhese suckers to the beginning and start the "regular flow"
        # I think maybe use linear motion b/c I'll want to move the network to the bottom right 
        # probably at the same time. 

        # Hmm reading 24 kind feeling like I want to maybe show some equatoins- probably just from illustrator
        # and maybe on the bottom of the screen? Let me hack on some layouts. 
        group_11=Group(surface_21, polygons_21b, axes_1, joint_line_11)
        group_12=Group(surface_22, polygons_22b, axes_2, joint_line_12)



        # Ok this kinda stucks, but I think that the top and bottom panels need ot be their own axes/scenes?
        # Can't seem to get and angle that works for both plots like this...
        # Ok so I think I need two different moves/scenes for this, and that's ok -> 
        # And also on this move, pretty sure I need to reverse the fold so we can play it again as the data moves through the 
        # network!!
        # group_11.move_to([0, 0, 0.8])
        # group_12.move_to([0, 0, -0.8])
        # self.frame.reorient(0, 66, 0, (-0.06, -0.02, 0.06), 3.82)

        surface_11_copy.shift([0, 0, -0.9])
        polygons_21a_copy.shift([0, 0, -0.9])
        polygons_11_copy.shift([0, 0, -0.9])
        surface_12_copy.shift([0, 0, 0.9])
        polygons_22a_copy.shift([0, 0, 0.9])
        polygons_12_copy.shift([0, 0, 0.9])        
        # self.remove(group_12) #Eh i think no: Animate this move in a seperate scene, bring together in editing. 

        self.wait()
        self.play(self.frame.animate.reorient(0, 58, 0, (0.19, -0.12, -0.1), 3.82),
                   axes_1.animate.shift([0, 0, -0.9]),
                   joint_line_11.animate.shift([0, 0, -0.9]), #.set_opacity(0.0),
                   ReplacementTransform(polygons_21b, polygons_21a_copy),
                   ReplacementTransform(surface_21, surface_11_copy),
                   axes_2.animate.shift([0, 0, 0.9]),
                   joint_line_12.animate.shift([0, 0, 0.9]), #.set_opacity(0.0),
                   ReplacementTransform(polygons_22b, polygons_22a_copy),
                   ReplacementTransform(surface_22, surface_12_copy),
                  run_time=4)
        self.remove(surface_11_copy); self.add(surface_11_copy)
        self.remove(polygons_21a_copy); self.add(polygons_11_copy)
        self.remove(axes_1); self.add(axes_1)
        self.remove(surface_12_copy); self.add(surface_12_copy)
        self.remove(polygons_22a_copy); self.add(polygons_12_copy)
        self.remove(axes_2); self.add(axes_2)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(joint_line_12); self.add(joint_line_12)
        self.wait()

        # Ok i think that will work pretty well. 
        # Fighting with overall composition a little - probably 3 panels at the end, right?
        # I'm going to work on equations next for a bit. 
        # Ok i think i got a good solution for the first 2 equations!


        # Alright now how to i put map points on the surface? 
        # And maybe add 3d guidelines? Let me look at how I did this last time...
        # Ok p35_41.mp4 has that stuuuf
        # And 46 is also helpful -> it has all the panels for this model!
        # Ok so i think first let me just try to get a magenta 3d sphere onto these maps!

        # self.frame.reorient(0, 58, 0, (0.19, -0.12, -0.1), 3.82),

        d1=Dot(ORIGIN, radius=0.04, fill_color=MAGENTA)
        d1.move_to(axes_1.c2p(0.59, 0.4, -0.04))
        d1.rotate(-20*DEGREES, [0, 1, 0])

        d2=Dot(ORIGIN, radius=0.05, fill_color=MAGENTA)
        d2.move_to(axes_2.c2p(0.59, 0.4, -0.05))
        d2.rotate(20*DEGREES, [1, 0, 0])


        self.wait()
        # self.add(joint_line_11, , joint_line_12) #Just keep from before?
        self.play(self.frame.animate.reorient(-37, 56, 0, (0.59, -0.49, -0.13), 3.75), 
                 FadeIn(d1), 
                 FadeIn(d2), 
                 run_time=4, 
                 rate_func=linear #Linear rate func here to blend with Premiere move!
                 )
        self.wait()


        #Alright time to bend these puppies back up! I think bend them at the same time. 

        polygons_21a.shift([0, 0, -0.9])
        surface_21_copy.shift([0, 0, -0.9])
        polygons_22a.shift([0, 0, 0.9])
        surface_22_copy.shift([0, 0, 0.9])

        d1b=Dot(ORIGIN, radius=0.04, fill_color=MAGENTA)
        d1b.move_to(axes_1.c2p(0.59, 0.4, 0))
        # d1b.rotate(-20*DEGREES, [0, 1, 0])
        d2b=Dot(ORIGIN, radius=0.05, fill_color=MAGENTA)
        d2b.move_to(axes_2.c2p(0.59, 0.4, 0))
        # d2b.rotate(20*DEGREES, [1, 0, 0])


        self.wait()
        self.remove(polygons_11_copy); self.add(polygons_21a_copy)
        self.play(ReplacementTransform(polygons_21a_copy, polygons_21a), 
                  ReplacementTransform(surface_11_copy, surface_21_copy),
                  ReplacementTransform(d1, d1b),
                 run_time=3.5)

        self.remove(polygons_21a); self.add(polygons_21a)
        self.remove(axes_1); self.add(axes_1)
        self.remove(joint_line_11); self.add(joint_line_11)
        self.remove(d1b); self.add(d1b)
        self.wait()

        #Eh maybe seqeuntial works best w/ directing focus and the script?
        self.remove(polygons_12_copy); self.add(polygons_22a_copy)
        self.play(ReplacementTransform(polygons_22a_copy, polygons_22a), 
                  ReplacementTransform(surface_12_copy, surface_22_copy),
                  ReplacementTransform(d2, d2b),
                 run_time=3.5)

        self.remove(polygons_22a); self.add(polygons_22a)
        self.remove(axes_2); self.add(axes_2)
        self.remove(joint_line_12); self.add(joint_line_22)
        self.remove(d2b); self.add(d2b)
        self.wait()




        # self.add(surface_21_copy)
        # self.add(surface_22_copy)

        
        # self.remove(surface_21_copy)

        # surface_21.set_opacity(1.0)

        # self.add(surface_21)
        # self.remove(surface_21)

        # self.add(polygons_21b)
        # self.remove(polygons_21b)

        # self.add(polygons_21a)
        # self.remove(polygons_21a)

        # # polygons_21b.shift([0, 0, 0.9])
        # self.add(polygons_21b)
        # self.remove(polygons_21b)




        # self.add(d1)

        # self.add(joint_line_11)
        # joint_line_11.set_opacity(1.0)

        # self.add(d2)

        # self.add(joint_line_12)
        # joint_line_12.set_opacity(1.0)

        # self.frame.reorient(-37, 56, 0, (0.59, -0.49, -0.13), 3.75)
        # self.wait()





        #Maybe I just do the separate moves in the same scene?
        # self.remove(group_11); self.remove(polygons_11_copy); self.remove(surface_11_copy)
        # self.wait()


        # self.frame.reorient(-53, 68, 0, (0.01, -0.1, 0.09), 5.58)
        # self.add(group_12)

        # # Ok cool now similiar move to above, just with different surfaces, and 
        # # end up at a little different camera angle!
        # surface_12_copy.shift([0, 0, 0.9])
        # polygons_22a_copy.shift([0, 0, 0.9])
        # polygons_12_copy.shift([0, 0, 0.9])

        # self.wait()
        # self.play(self.frame.animate.reorient(0, 83, 0, (-0.06, -0.06, 0.0), 3.82),
        #            axes_2.animate.shift([0, 0, 0.9]),
        #            joint_line_12.animate.shift([0, 0, 0.9]).set_opacity(0.0),
        #            ReplacementTransform(polygons_22b, polygons_22a_copy),
        #            ReplacementTransform(surface_22, surface_12_copy),
        #           run_time=4)
        # self.remove(surface_12_copy); self.add(surface_12_copy)
        # self.remove(polygons_22a_copy); self.add(polygons_12_copy)
        # self.remove(axes_2); self.add(axes_2)
        # self.wait()

        # Eh reading the script again I'm not so sure 
        # I think i should keep them locked to the same FoV if I can
        # Let me push forward on that path!







        # self.frame.reorient(0, 59, 0, (-0.06, -0.06, 0.0), 3.82)
        # self.add(group_11)

        # self.add(surface_12, polygons_22a, axes_2, joint_line_12)

        
        # self.add(surface_21, polygons_21b, axes_1, joint_line_11)
        # self.add(surface_22, polygons_22b, axes_2, joint_line_12)


        # self.wait()

        # self.add(surface_11, polygons_11)
        # # self.add(polygons_11)
        # self.add(surface_12, polygons_12)
        # self.add(axes_1, axes_2)
        # self.add(joint_line_11)
        # self.add(joint_line_12)

        # self.wait()
        
        #Ok great things are aligned - now I'm not going to bring in the map right away I think! Eh actually it helps alot. 
        #Ok so I'm getting close to starting working out the moves etc -> but I think makes sense to go ahead 
        #And get the bent planes in place, and test the folding transition
        #Then I can align everything to the script 







        self.wait(20)
        self.embed()













        # surfaces[0][0].move_to([0, 0, 1.7])

        # self.add(surfaces[0][0])






        # group_11.move_to([0, 0, 1.5])
        # group_12.move_to([0, 0, -1.5])



        # self.add(group_11)
        # self.add(group_12)


        # self.wait()







        # group_11.shift([0, 0, 1.5])






        # self.wait()









