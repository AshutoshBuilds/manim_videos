from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes


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

class p35_41(InteractiveScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl'
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        # train_step=2400
        train_step=7 #Start on step 7, this has the example I want
        w1=p['weights_history'][train_step]['model.0.weight'].numpy()
        b1=p['weights_history'][train_step]['model.0.bias'].numpy()
        w2=p['weights_history'][train_step]['model.2.weight'].numpy()
        b2=p['weights_history'][train_step]['model.2.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))


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
            polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
            polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
            polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        #Get first layer Relu Joints - 
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        group_11=Group(surfaces[1][0])
        if len(joint_points_11)>0:
            joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
            group_11.add(joint_line_11)

        group_12=Group(surfaces[1][1])
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        if len(joint_points_12)>0:
            joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
            group_12.add(joint_line_12)

        group_13=Group(surfaces[1][2])
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        if len(joint_points_13)>0:
            joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
            group_13.add(joint_line_13)

        group_11.shift([0, 0, 1.5])
        group_13.shift([0, 0, -1.5])

        polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_21.shift([0, 0, 0.001]) #Move slightly above map

        polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
        polygons_22.shift([0, 0, 0.001]) #Move slightly above map

        group_21=Group(surfaces[2][0], polygons_21)
        group_22=Group(surfaces[2][1], polygons_22)
        group_21.shift([3.0, 0, 0.6])
        group_22.shift([3.0, 0, -0.6])
        group_21.set_opacity(0.5)
        group_22.set_opacity(0.5)


        #Ok now some plane intersction action in a third "panel"
        group_31=group_21.copy()
        group_31[1].set_color(BLUE)
        group_31.shift([3, 0, -0.6])

        group_32=group_22.copy()
        group_32[1].set_color(YELLOW)
        group_32.shift([3, 0, 0.6])

        lines=VGroup()
        for loop in intersection_lines: 
            loop=loop*np.array([1, 1, viz_scales[2]])
            line = VMobject()
            line.set_points_as_corners(loop)
            line.set_stroke(color='#FF00FF', width=5)
            lines.add(line)
        lines.shift([6, 0, 0])

        #Make Baarle hertog maps a little mroe pronounced. 
        group_31[0].set_opacity(0.9)
        group_32[0].set_opacity(0.9)

        # group_21.set_opacity(0.9)
        # group_22.set_opacity(0.9)

        self.frame.reorient(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
        self.add(group_12, group_13)
        self.add(group_21, group_22)
        self.add(group_31, group_32, lines)
        self.add(group_11)


        #Hmm i feel like some kinda legend or labelling might be helpful here! Minimal i think is ok, but something
        # I can think on that next. Ok done did it in illustrator. 
        # Now is it worth adding a small final decision bounday ochear kidna deal? 
        # Eh that's going to be a mess in 3d. Let my try making maps a little more proncounced in final surface here. 
        # Alright time to start moving the example point through the model!
        # Can already tell that I probalby want a more detailed play by play in the narration. 

        map_coords_1=Tex(r'(0.6, 0.4)', font_size=11).set_color('#FF00FF')
        map_pt_1=Dot(ORIGIN, radius=0.015).set_color('#FF00FF')
        map_pt_1.move_to([-0.39, 0.12, 0])
        map_coords_1.move_to([-0.4, 0.04, 0])
        coords_group_1=VGroup(map_pt_1, map_coords_1)
        coords_group_1.scale(2.0)
        coords_group_1.move_to([0.56,0.22,1.5])

        coords_group_2=coords_group_1.copy()
        coords_group_3=coords_group_1.copy()
        coords_group_2.move_to([0.56,0.22,0.0])
        coords_group_3.move_to([0.56,0.22,-1.5])

        coords_group_4=coords_group_2.copy()
        coords_group_5=coords_group_2.copy()
        coords_group_4.shift([3,0.0,0.77])
        coords_group_5.shift([3,0.0,-0.83])

        #Zoom in to overhead of first map and draw on point!
        self.wait()


        # self.add(group_11) #Do i need to fix occlusions again or no? 
        self.play(self.frame.animate.reorient(0, 3, 0, (-0.01, 0.19, -0.42), 4.81), 
                  group_11.animate.set_opacity(1.0),
                 run_time=4)
        self.wait()

        #Now write on points!
        self.play(Write(coords_group_1[1]), FadeIn(coords_group_1[0]), run_time=2)
        self.wait()

        self.play(self.frame.animate.reorient(-1, 47, 0, (3.25, 0.26, -0.22), 4.15), 
                  FadeIn(coords_group_2[0]), #Try just the dot
                  FadeIn(coords_group_3[0]),
                  FadeIn(coords_group_5[0]),
                  FadeIn(coords_group_4[0]),
                  Write(coords_group_4[1]), #But include coords on top layer!
                 run_time=5.0)
        self.wait()

        # Oh man could I do a cool highlight of the linear region I'm talking about?!
        # self.play(Indicate(polygons_21[0], color=GREY))
        # Oh man could I do a cool highlight of the linear region I'm talking about?!
        outline = polygons_21[0].copy()
        outline.set_fill(opacity=0)
        outline.set_stroke('#FF00FF', width=4, opacity=0.9)
        self.play(ShowCreation(outline, run_time=2))
        self.wait()

        #Ok ok ok I think i want to go overhead, may switch to top polygons, and for bonus points actually REplaceTransformt eh called
        # out polygon and label! Than I can add netherlands belgium labels in illustrator. 

        # Ok let me think ahead for a minute first though -> I want to show a single gradient descent step
        # And I do think that showing all the actually gradient values on the network would be pretty cool!
        # Anyway though, I need some example that actually makes sense lol -> Let me render out a loop of the first N steps 
        # And see if I can make sense of what's going on -> I may need to combine steps too -> I think that would be ok!



        # self.add(coords_group_5)
        # self.add(coords_group_4)
        # self.add(coords_group_2)
        # self.add(coords_group_3)

        # Ok I think a starting step of 7, then a step size of 5 can make for a nice viz, and I'll focus on the bottom 
        # first plane getting less negative steep - I think that makes sense - just gotta work through the visuals, work on the 
        # Script a bit, then play the whole thing - oh yeah and I need to finish the last part of the move above. 
        # Complicated but cool/important scene I think!
        



        # self.add(coords_group_1)
        # self.add(coords_group_2)
        # self.add(coords_group_3)

        self.wait()


        

class p36_loop_test_b(InteractiveScene):
    def construct(self):

        model = BaarleNet([3])
        pickle_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/3_starting_configs_4/training_data_seed_01_acc_0.8561.pkl'
        with open(pickle_path, 'rb') as f:
            p=pickle.load(f)

        print(len(p['weights_history']), ' training steps loaded.')

        viz_scales=[0.2, 0.2, 0.1]
        num_neurons=[3, 3, 2]

        self.frame.reorient(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)

        #Ok having trouble reproducing this training config, so we're going to grab a straring point that 
        # has the same label and is in the same neighborhood!

        # train_step=2400
        start_step=7 #Start here to get the gradietns I want to show
        step_size=5

        print('starting grads', p['gradients_history'][start_step])
        print('empirical grads w1: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-
                                           p['weights_history'][7]['model.0.weight'].numpy())) #-1/lr (0.01)
        print('empirical grads w2: ', -100*(p['weights_history'][8]['model.2.weight'].numpy()-
                                           p['weights_history'][7]['model.2.weight'].numpy())) #-1/lr (0.01)
        print('empirical grads b1: ', -100*(p['weights_history'][8]['model.0.bias'].numpy()-
                                           p['weights_history'][7]['model.0.bias'].numpy())) #-1/lr (0.01)
        print('empirical grads b2: ', -100*(p['weights_history'][8]['model.2.bias'].numpy()-
                                           p['weights_history'][7]['model.2.bias'].numpy())) #-1/lr (0.01)


# In [5]: print('empirical grads: ', -100*(p['weights_history'][8]['model.0.weight'].numpy()-p['weights_history'][7]['model.0.weight'].numpy()))                                      
# empirical grads:  [[ 0.38974938 -0.45838356]                                                                                                                                        
#  [ 0.16942024 -0.63158274]                                                                                                                                                          
#  [-1.0046124   0.9809494 ]]                                                                                                                                                         
# In [6]: print('starting grads', p['gradients_history'][start_step])                                                                                                                 
# starting grads {'model.0.weight': tensor([[ 0.0021, -0.0006],                                                                                                                       
#         [-0.0005,  0.0004],                                                                                                                                                         
#         [-0.3566,  0.2963]]), 'model.0.bias': tensor([-0.0006, -0.0005, -0.4256]), 'model.2.weight': tensor([[-0.0074, -0.0003, -0.0616],                                           
#         [ 0.0074,  0.0003,  0.0616]]), 'model.2.bias': tensor([ 0.2037, -0.2037])} 

        #Ok I think there's some issue with my analytical gradients -> updates don't seem to match -> 
        # For numbers on screen I'll show empirical grads then




        for train_step in np.arange(start_step, 50, step_size):
            if 'group_12' in locals():
                self.remove(group_12, group_13)
                self.remove(group_21, group_22)
                self.remove(group_31, group_32, lines)
                self.remove(group_11)

            w1=p['weights_history'][train_step]['model.0.weight'].numpy()
            b1=p['weights_history'][train_step]['model.0.bias'].numpy()
            w2=p['weights_history'][train_step]['model.2.weight'].numpy()
            b2=p['weights_history'][train_step]['model.2.bias'].numpy()

            with torch.no_grad():
                model.model[0].weight.copy_(torch.from_numpy(w1))
                model.model[0].bias.copy_(torch.from_numpy(b1))
                model.model[2].weight.copy_(torch.from_numpy(w2))
                model.model[2].bias.copy_(torch.from_numpy(b2))


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
                polygons[str(layer_id)+'.split_polygons_nested']=split_polygons_with_relu_simple(polygons[str(layer_id)+'.linear_out']) #Triple nested list so we can simplify merging process layer. 
                polygons[str(layer_id)+'.split_polygons_nested_clipped'] = clip_polygons(polygons[str(layer_id)+'.split_polygons_nested'])
                polygons[str(layer_id)+'.split_polygons_merged'] = merge_zero_regions(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
                polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
                print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')

            #Last linear layer & output
            polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
            intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
            my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


            #Get first layer Relu Joints - 
            joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
            group_11=Group(surfaces[1][0])
            if len(joint_points_11)>0:
                joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.9)
                group_11.add(joint_line_11)

            group_12=Group(surfaces[1][1])
            joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
            if len(joint_points_12)>0:
                joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.9)
                group_12.add(joint_line_12)

            group_13=Group(surfaces[1][2])
            joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
            if len(joint_points_13)>0:
                joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.9)
                group_13.add(joint_line_13)

            group_11.shift([0, 0, 1.5])
            group_13.shift([0, 0, -1.5])

            polygons_21=manim_polygons_from_np_list(polygons['1.linear_out'][0], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_21.shift([0, 0, 0.001]) #Move slightly above map

            polygons_22=manim_polygons_from_np_list(polygons['1.linear_out'][1], colors=colors, viz_scale=viz_scales[2], opacity=0.6)
            polygons_22.shift([0, 0, 0.001]) #Move slightly above map

            group_21=Group(surfaces[2][0], polygons_21)
            group_22=Group(surfaces[2][1], polygons_22)
            group_21.shift([3.0, 0, 0.6])
            group_22.shift([3.0, 0, -0.6])
            group_21.set_opacity(0.5)
            group_22.set_opacity(0.5)


            #Ok now some plane intersction action in a third "panel"
            group_31=group_21.copy()
            group_31[1].set_color(BLUE)
            group_31.shift([3, 0, -0.6])

            group_32=group_22.copy()
            group_32[1].set_color(YELLOW)
            group_32.shift([3, 0, 0.6])

            lines=VGroup()
            for loop in intersection_lines: 
                loop=loop*np.array([1, 1, viz_scales[2]])
                line = VMobject()
                line.set_points_as_corners(loop)
                line.set_stroke(color='#FF00FF', width=5)
                lines.add(line)
            lines.shift([6, 0, 0])

            #Make Baarle hertog maps a little mroe pronounced. 
            group_31[0].set_opacity(0.9)
            group_32[0].set_opacity(0.9)

            # group_21.set_opacity(0.9)
            # group_22.set_opacity(0.9)

            # self.frame.reorient(-1, 46, 0, (3.09, 0.56, -0.42), 7.25)
            self.add(group_12, group_13)
            self.add(group_21, group_22)
            self.add(group_31, group_32, lines)
            self.add(group_11)
            self.wait(0.1)



            

        self.wait(20)
        self.embed()
