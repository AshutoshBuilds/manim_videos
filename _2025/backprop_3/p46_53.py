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

        # self.frame.reorient(-1, 42, 0, (3.1, 0.59, -0.39), 6.92)
        # self.frame.reorient(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)

        # self.wait()
        # self.play(FadeIn(group_11), FadeIn(group_12))
        # self.wait()

        # self.play(FadeIn(group_21), FadeIn(group_22)) #Would be nice to animate layer 1 neurons coming together instead - we'll see. 
        # self.wait()

        bent_plane_joint_lines=VGroup()
        pre_move_lines=VGroup()

        line_start=polygons['1.linear_out'][0][1][1]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][1][2]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][1][1]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][1][2]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][0][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 0])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][0]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        self.frame.reorient(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)
        self.wait()
        self.play(FadeIn(group_11[0]), FadeIn(group_12[0]), FadeIn(pre_move_lines))
        self.wait()


        # self.add(joint_line)  
        self.wait()
        self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
                  ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
                  ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines), 
                    run_time=3)
        self.add(polygons_21)
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        bent_plane_joint_lines_2=VGroup()
        pre_move_lines_2=VGroup()

        line_start=polygons['1.linear_out'][1][1][1]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][1][2]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][1][1]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][1][2]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][0][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 0])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        line_start=polygons['1.linear_out'][1][2][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0.0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][0]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        self.wait()
        self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][1]),
                  ReplacementTransform(surfaces[1][1].copy(),surfaces[2][1]),
                  ReplacementTransform(pre_move_lines_2.copy(), bent_plane_joint_lines_2), 
                  run_time=3)
        self.add(polygons_22)
        self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        self.wait()


        #Ok now a little animation bringin the two bent surfaces together and changing their colors? 
        self.play(ReplacementTransform(group_21.copy(), group_31), 
                  ReplacementTransform(group_22.copy(), group_32), 
                 run_time=3.0)
        self.play(ShowCreation(lines))
        self.wait()

        #Tempted to zoom in on decision boundary, but mabye we've seen it enough?
        # Ok now need ot hink through how this fits with illustrator network and then exapnding it! Good start!
        #Ok will do a new scene for the 2 layer model!


        # self.add(group_31, group_32, lines)
        # self.wait()

        # self.play(ReplacementTransform(polygons_11[0].copy(), polygons_21[2:]))


        self.wait(20)
        self.embed()


class p47(InteractiveScene):
    def construct(self):

        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        map_img.set_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.set_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(ORIGIN)


        model_path='_2025/backprop_3/models/2_2_1.pth'
        model = BaarleNet([2,2])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        num_neurons=[2, 2, 2, 2, 2]

        w1=model.model[0].weight.detach().numpy()
        b1=model.model[0].bias.detach().numpy()
        w2=model.model[2].weight.detach().numpy()
        b2=model.model[2].bias.detach().numpy()

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
        group_31_fakeout=group_21.copy()
        group_31_fakeout[1].set_color(BLUE)
        group_31_fakeout.shift([3, 0, -0.75])

        group_32_fakeout=group_22.copy()
        group_32_fakeout[1].set_color(YELLOW)
        group_32_fakeout.shift([3, 0, 0.75])

        # loops=order_closed_loops_with_closure(intersection_lines)
        # lines_fakeout=VGroup()
        # for loop in loops: 
        #     loop=loop*np.array([1, 1, viz_scales[2]])
        #     line = VMobject()
        #     line.set_points_as_corners(loop)
        #     line.set_stroke(color='#FF00FF', width=5)
        #     lines_fakeout.add(line)
        # lines_fakeout.shift([6, 0, 0.75])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31_fakeout[0].set_opacity(0.9)
        group_32_fakeout[0].set_opacity(0.9)
        group_31_fakeout[1].set_opacity(0.4)
        group_32_fakeout[1].set_opacity(0.4)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        # self.frame.reorient(-1, 42, 0, (3.1, 0.59, -0.39), 6.92)
        bent_plane_joint_lines=VGroup()
        pre_move_lines=VGroup()

        line_start=polygons['1.linear_out'][0][1][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][1][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][1][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][1][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][0]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][0]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][2][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][0][0][3]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 1.5])
        bent_plane_joint_lines.add(joint_line)

        line_start=polygons['1.linear_out'][0][0][2]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][0][0][3]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines.add(joint_line)


        self.frame.reorient(-2, 46, 0, (3.11, 0.22, -0.73), 6.89)

        self.wait()
        self.play(FadeIn(group_11[0]), FadeIn(group_12[0]), FadeIn(pre_move_lines))
        self.wait()

        # Ok I think it's imporant here (and maybe in p46 above) to actually do the brining the 
        # lines together animation, as annoying as it is
        # It's only 2 lines - I can do it -> and I've found a way to make it work -> it's just annoying!

        # self.remove(group_11[1]); self.remove(group_12[1]) #Remove existing Relu lines to replace with "premove lines"
        # self.add(pre_move_lines)


        # self.add(joint_line)  
        self.wait()
        self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][0]),
                  ReplacementTransform(surfaces[1][1].copy(),surfaces[2][0]),
                  ReplacementTransform(pre_move_lines.copy(), bent_plane_joint_lines), 
                    run_time=3)
        self.add(polygons_21)
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        bent_plane_joint_lines_2=VGroup()
        pre_move_lines_2=VGroup()

        line_start=polygons['1.linear_out'][1][1][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][1][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][1][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][1][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][3]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][0]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][3]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][0]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][2][0]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][2][1]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][2]*np.array([1,1,viz_scales[2]])
        line_end=polygons['1.linear_out'][1][0][3]*np.array([1,1,viz_scales[2]])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        joint_line.shift([3, 0, 0])
        bent_plane_joint_lines_2.add(joint_line)

        line_start=polygons['1.linear_out'][1][0][2]*np.array([1,1,0]) #z=0 version
        line_end=polygons['1.linear_out'][1][0][3]*np.array([1,1,0])
        joint_line = DashedLine(start=line_start, end=line_end, color=WHITE, stroke_width=3, dash_length=0.05)
        # joint_line.shift([0, 0, 1.5])
        pre_move_lines_2.add(joint_line)


        self.wait()
        self.play(ReplacementTransform(surfaces[1][0].copy(),surfaces[2][1]),
                  ReplacementTransform(surfaces[1][1].copy(),surfaces[2][1]),
                  ReplacementTransform(pre_move_lines_2.copy(), bent_plane_joint_lines_2), 
                  run_time=3)
        self.add(polygons_22)
        self.remove(bent_plane_joint_lines_2); self.add(bent_plane_joint_lines_2)
        self.wait()

        #Ok now quick "fake out 3 layer", then remove everything from that. 
        self.play(ReplacementTransform(group_21.copy(), group_31_fakeout), 
                  ReplacementTransform(group_22.copy(), group_32_fakeout), 
                 run_time=3.0)
        # self.play(ShowCreation(lines_fakeout))
        self.wait()
        self.play(FadeOut(group_31_fakeout), FadeOut(group_32_fakeout))
        self.wait()

        # I'm a bit torn on working off a copy to the right vs doing this in place -> 
        # I guess I'm leaning towards doing it in place?
        # Ok I think we add z=0 planes, actually let's focus/zoom on top neuron, and add the plane there
        relu_intersections_planes_1=VGroup()
        for neuron_idx in range(2):
            # plane = Rectangle( width=2,  height=2, fill_color=GREY, fill_opacity=0.3, stroke_color=WHITE, stroke_width=2)
            plane = NumberPlane(
                x_range=(-1, 1, 0.2),  # x from -1 to 1, grid every 0.2 units
                y_range=(-1, 1, 0.2),  # y from -1 to 1, grid every 0.2 units
                background_line_style={
                    "stroke_color": WHITE,
                    "stroke_width": 0.5,
                    "stroke_opacity": 0.9
                },
                faded_line_style={
                    "stroke_color": WHITE,
                    "stroke_width": 0.0,
                    "stroke_opacity": 0.0
                },
                axis_config={
                    "stroke_color": WHITE,
                    "stroke_width": 0.5
                }
            ).set_width(2).set_height(2)
            plane.shift([3, 0, 1.5*neuron_idx])
            relu_intersections_planes_1.add(plane)
        
        self.wait()
        self.play(self.frame.animate.reorient(30, 70, 0, (2.55, 1.16, 0.93), 4.27), 
                  group_11.animate.set_opacity(0.2), 
                  group_12.animate.set_opacity(0.2),
                  pre_move_lines.animate.set_opacity(0.0),
                  run_time=3)
        self.play(ShowCreation(relu_intersections_planes_1[1]))
        self.wait()

        self.play(self.frame.animate.reorient(32, 83, 0, (2.55, 1.16, 0.93), 4.27), run_time=3)
        self.wait()

        #Outline to call out planes, I can cut this if I don't like it. 
        outline = polygons_21[1].copy()
        outline.set_fill(opacity=0)
        outline.set_stroke(width=4, opacity=0.9)
        self.play(ShowCreation(outline, run_time=2))
        # self.play(FadeOut(outline))

        outline_2 = polygons_21[3].copy()
        outline_2.set_fill(opacity=0)
        outline_2.set_stroke(width=4, opacity=0.9)
        self.play(ShowCreation(outline_2), FadeOut(outline), run_time=2)

        outline_3 = polygons_21[2].copy()
        outline_3.set_fill(opacity=0)
        outline_3.set_stroke(width=4, opacity=0.9)
        self.play(ShowCreation(outline_3), FadeOut(outline_2), run_time=2)
        self.play(FadeOut(outline_3))
        self.remove(bent_plane_joint_lines); self.add(bent_plane_joint_lines)
        self.wait()

        # Ok ok ok now how do I animate folidng this surface up, and what's a good camera angle for it?
        # In the middle of p49. 
        self.play(self.frame.animate.reorient(33, 61, 0, (2.54, 0.96, 0.38), 4.24), run_time=2)

        surfaces[3][0].shift([3, 0, 1.5])

        # split_polygons_merged
        # split_polygons_nested_clipped
        #Maybe i transform to split_polygons_nested_clipped, and then once we're flat then merge to split_polygons_merged
        
        split_polygons_unraveled=[item for sublist in polygons['1.split_polygons_nested'][0] for item in sublist]
        split_polygons_unraveled_clipped=[item for sublist in polygons['1.split_polygons_nested_clipped'][0] for item in sublist] #Need to unravel this i think? And then try to animate to it? 

        colors_2 = [GREY, BLUE, BLUE, GREEN, GREEN, YELLOW, YELLOW]
        colors_3 = [GREY, BLUE, GREEN, YELLOW]
        polygons_31=manim_polygons_from_np_list(split_polygons_unraveled, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31.shift([3, 0, 1.501]) #Move slightly above map

        polygons_31_clipped=manim_polygons_from_np_list(split_polygons_unraveled_clipped, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_clipped.shift([3, 0, 1.501]) #Move slightly above map

        polygons_31_merged=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][0], colors=colors_3, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_merged.shift([3, 0, 1.501]) #Move slightly above map

        

        # Man this polygon transform is secretly kinda tricky lol. 
        # What if I grabbed the pre-clipped polygons, and colored them to  match, then moved them? 
        # Ok yeah I think that's the move! Yeah we just swap out the split ones that aren't clipped rigth befor the move I think
        #Ok yeah that's the motion I want -> I'll dropped the dashed lines right before too I think 

        self.wait()
        self.remove(polygons_21); self.add(polygons_31); self.remove(bent_plane_joint_lines)
        self.play(ReplacementTransform(polygons_31, polygons_31_clipped),
                  ReplacementTransform(surfaces[2][0], surfaces[3][0]), 
                  FadeOut(relu_intersections_planes_1[1]),
                 run_time=3)
        self.remove(polygons_31_clipped); self.add(polygons_31_clipped)
        self.wait()
  
        # self.play(ReplacementTransform(VGroup(*[polygons_31_clipped[i] for i in [0,2,4,6]]).copy(), polygons_31_merged[0]))

        self.remove(polygons_31_clipped[1], polygons_31_clipped[3], polygons_31_clipped[5])
        self.add(polygons_31_merged[1:]) #, polygons_31_merged[3], polygons_31_merged[5])
        self.play(FadeOut(VGroup(*[polygons_31_clipped[i] for i in [0,2,4,6]])), FadeIn(polygons_31_merged[0])) #Nice!
        self.wait()

        self.play(self.frame.animate.reorient(-1, 38, 0, (3.12, 1.06, 0.18), 4.24), run_time=6)
        self.wait()

        #Ok now second layer. 
        self.play(FadeIn(relu_intersections_planes_1[0]), 
                  self.frame.animate.reorient(-132, 58, 0, (2.41, 0.82, 0.12), 2.31), 
                  run_time=4)
        self.wait()

        surfaces[3][1].shift([3, 0, 0])
        split_polygons_unraveled_2=[item for sublist in polygons['1.split_polygons_nested'][1] for item in sublist]
        split_polygons_unraveled_clipped_2=[item for sublist in polygons['1.split_polygons_nested_clipped'][1] for item in sublist] #Need to unravel this i think? And then try to animate to it? 

        polygons_32=manim_polygons_from_np_list(split_polygons_unraveled_2, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32.shift([3, 0, 0.001]) #Move slightly above map

        polygons_32_clipped=manim_polygons_from_np_list(split_polygons_unraveled_clipped_2, colors=colors_2, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_clipped.shift([3, 0, 0.001]) #Move slightly above map

        colors_4 = [GREY, GREY, BLUE, GREEN, YELLOW]
        polygons_32_merged=manim_polygons_from_np_list(polygons['1.split_polygons_merged'][1], colors=colors_4, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_merged.shift([3, 0, 0.001]) #Move slightly above map

        self.wait()
        self.remove(polygons_22); self.add(polygons_32); self.remove(bent_plane_joint_lines_2)
        self.play(ReplacementTransform(polygons_32, polygons_32_clipped),
                  ReplacementTransform(surfaces[2][1], surfaces[3][1]), 
                  FadeOut(relu_intersections_planes_1[0]),
                 run_time=3)
        self.remove(polygons_32_clipped); self.add(polygons_32_clipped)
        self.wait()

        self.remove(polygons_32_clipped[1], polygons_32_clipped[3], polygons_32_clipped[5], polygons_32_clipped[0])
        self.add(polygons_32_merged[1:]) #, polygons_31_merged[3], polygons_31_merged[5])
        self.play(FadeOut(VGroup(*[polygons_32_clipped[i] for i in [2,4,6]])), FadeIn(polygons_32_merged[0])) #Nice!
        self.wait()


        self.play(self.frame.animate.reorient(0, 57, 0, (2.72, 0.7, -0.61), 6.70), 
                  group_11.animate.set_opacity(0.8), 
                  group_12.animate.set_opacity(0.8), run_time=6)
        self.wait()

        #Ok now cool 2d collapsing down to tiling for first layer and then second, let's go!

        #Go ahead and establish enpoints and then figure out how we want to get there!
        layer_1_polygons_flat=manim_polygons_from_np_list(polygons['0.new_tiling'], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        layer_1_polygons_flat.shift([0, 0, -1.5])

        colors_5=[GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL, MAROON, GREEN_B]
        layer_2_polygons_flat=manim_polygons_from_np_list(polygons['1.new_tiling'], colors=colors_5, viz_scale=viz_scales[2], opacity=0.6)
        layer_2_polygons_flat.shift([3, 0, -1.5])

        #Ok bring down copies of the Relu folds? maps too or nah
        def flat_surf_func(u, v): return [u, v, 0]
        flat_map_surf = ParametricSurface(flat_surf_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        flat_map=TexturedSurface(flat_map_surf, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-17.png')
        flat_map.set_shading(0,0,0).set_opacity(0.8)
        flat_map.shift([0, 0, -1.5])

        fold_copy_1=group_11[1].copy()
        fold_copy_2=group_12[1].copy()
        self.wait()
        self.play(fold_copy_1.animate.shift([0,0,-3]), 
                  fold_copy_2.animate.shift([0,0,-1.5]), 
                  ReplacementTransform(group_11[0].copy(), flat_map),
                  ReplacementTransform(group_12[0].copy(), flat_map),
                  run_time=3) 
        self.add(layer_1_polygons_flat)
        self.remove(fold_copy_1); self.add(fold_copy_1)
        self.remove(fold_copy_2); self.add(fold_copy_2)
        self.remove(flat_map)
        self.wait()

        # Ok so for projecting down the second layer stuff, how about just bringing down the outlines? 
        # I think that could be cool!

        outline_4 = polygons_31_merged.copy()
        outline_4.set_fill(opacity=0)
        # outline_4.set_stroke(width=4, opacity=0.9)
        outline_5 = polygons_32_merged.copy()
        outline_5.set_fill(opacity=0)
        # outline_5.set_stroke(width=4, opacity=0.9)
        
        ##Hmm I need a flat version of these down on the bottom to move to. 
        ## Might need to create another little set of polygons?
        polygons_31_merged_flat_arrays=polygons['1.split_polygons_merged'][0]
        for o in polygons_31_merged_flat_arrays: o[:,2]=0 #Flatten that shit
        polygons_31_merged_flat=manim_polygons_from_np_list(polygons_31_merged_flat_arrays, colors=colors_3, viz_scale=viz_scales[2], opacity=0.6)
        polygons_31_merged_flat.set_fill(opacity=0)
        polygons_31_merged_flat.shift([3, 0, -1.5])

        polygons_32_merged_flat_arrays=polygons['1.split_polygons_merged'][1]
        for o in polygons_32_merged_flat_arrays: o[:,2]=0 #Flatten that shit
        polygons_32_merged_flat=manim_polygons_from_np_list(polygons_32_merged_flat_arrays, colors=colors_4, viz_scale=viz_scales[2], opacity=0.6)
        polygons_32_merged_flat.set_fill(opacity=0)
        polygons_32_merged_flat.shift([3, 0, -1.5])

        self.wait()
        self.play(ReplacementTransform(outline_4, polygons_31_merged_flat), run_time=4)
        self.play(ReplacementTransform(outline_5, polygons_32_merged_flat), run_time=3)
        self.wait()

        self.play(
                surfaces[3][1].animate.set_opacity(0.2),
                polygons_32_merged.animate.set_opacity(0.2),
                layer_1_polygons_flat.animate.set_opacity(0.2),
                self.frame.animate.reorient(0, 44, 0, (3.15, -0.12, -1.73), 3.42),
                run_time=4)
        self.wait()
        self.play(ShowCreation(layer_2_polygons_flat), run_time=3)
        self.wait()

        self.play(self.frame.animate.reorient(0, 60, 0, (2.86, 0.74, -0.34), 6.73), 
                surfaces[3][1].animate.set_opacity(0.6),
                polygons_32_merged.animate.set_opacity(0.8),
                layer_1_polygons_flat.animate.set_opacity(0.25), #Keep projections but don't make them a main focus. 
                layer_2_polygons_flat.animate.set_opacity(0.25),
                run_time=4)
        self.wait()

        #Ok now the classic merging things deal, and batch colors to the 2d projections!




        # self.add(polygons_31_merged_flat)
        # self.add(polygons_32_merged_flat)


        # self.add(layer_1_polygons_flat)
        # self.add(layer_2_polygons_flat)
        # self.remove(layer_2_polygons_flat)

        # self.add(polygons_31)
        # self.remove(polygons_31)

        # self.wait()
        
        # self.wait()
        # self.play(ReplacementTransform(surfaces[2][0], surfaces[3][0]), run_time=3)

        # self.add(bent_plane_joint_lines)
        # self.add(pre_move_lines)



        self.wait(20)
        self.embed()












