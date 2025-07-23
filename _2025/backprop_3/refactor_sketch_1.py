from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class refactor_sketch_1(InteractiveScene):
    def construct(self):
        # Ok I generally don't think refactoring on these projects is a good use of time
        # buuut this stuff is gettring pretty unwieldy, and I think Claude can write me a clean API quickly.
        # I'm realizing that there's really just a few operations that I repeate a bunch, so I want to get them uniform
        # I only have lik 9 days to finishe this fucker, but I think this will be worth it!
        # First change i want to make is just loading full torch models here - this will streamlien things
        
        #2x2
        # model_path='_2025/backprop_3/models/2_2_1.pth'
        # model = BaarleNet([2,2])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.25, 0.25, 0.2, 0.2, 0.15]
        # num_neurons=[2, 2, 2, 2, 2]

        #3x3
        model_path='_2025/backprop_3/models/3_3_1.pth'
        model = BaarleNet([3,3])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        num_neurons=[3, 3, 3, 3, 2]


        surfaces=[]
        surface_funcs=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                ts.set_shading(0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                self.add(s)

        # Ok looking good, tomorrow we tackle polygons!
        # Hmm ok yeah so i think the dashed fold line is nice early on in animtation
        # But here I guess at the very first relu (and you could even do before that),
        # I just want ot start figuing out zero crossing polygons.
        self.frame.reorient(0, 38, 0, (-0.16, 0.17, 0.1), 9.70)

        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)
        layer_1_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_1_polygons_3d.append([])
            for region in ['positive_region', 'negative_region']:
                a=[]
                for pt_idx in range(len(layer_1_polygons[neuron_idx][region])):
                    a.append(surface_funcs[layer_idx][neuron_idx](*layer_1_polygons[neuron_idx][region][pt_idx]))
                a=np.array(a)
                layer_1_polygons_3d[-1].append(a)

        #Now move to rigth locations and visualize polygons. 
        layer_1_polygons_vgroup=VGroup()
        layer_1_colors=[TEAL, PURPLE]
        for neuron_idx, polygons in enumerate(layer_1_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = Polygon(*p,
                                 fill_color=layer_1_colors[j],
                                 fill_opacity=0.7,
                                 stroke_color=layer_1_colors[j],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_1_polygons_vgroup.add(poly_3d)

        self.add(layer_1_polygons_vgroup)
        self.wait()

        # Ok, that's pretty dope! Now we need to merge these borders
        # Start in 2d again, then back up to 3d
        # I like the idea of a "collapsing down" animation to show the borders 
        # comign together - I can do that next time though!
        layer_idx=2
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])
        layer_2_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_2_polygons_3d.append([])
            for region in layer_2_polygons:
                a=[]
                for pt_idx in range(len(region)):
                    a.append(surface_funcs[layer_idx][neuron_idx](*region[pt_idx])) #Might be a batch way to do this
                a=np.array(a)
                layer_2_polygons_3d[-1].append(a)

        layer_2_polygons_vgroup=VGroup()
        layer_2_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
        for neuron_idx, polygons in enumerate(layer_2_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = Polygon(*p,
                                 fill_color=layer_2_colors[j],
                                 fill_opacity=0.7,
                                 stroke_color=layer_2_colors[j],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_2_polygons_vgroup.add(poly_3d)

        self.add(layer_2_polygons_vgroup)
        self.wait()

        #Ok nice!


        


        self.wait()

        

        self.wait()





        self.wait()
        self.embed()