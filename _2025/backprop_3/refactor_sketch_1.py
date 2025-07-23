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
        model_path='_2025/backprop_3/models/2_2_1.pth'
        model = BaarleNet([2,2])
        model.load_state_dict(torch.load(model_path))

        viz_scales=[0.25, 0.25, 0.2, 0.2, 0.15]

        surfaces=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            for neuron_idx in range(2): #Hardccoded righ tnow
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=viz_scales[layer_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                ts.set_shading(0,0,0).set_opacity(0.75)
                s.add(ts)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                self.add(s)


        #Ok looking good, tomorrow we tackle polygons!

        self.wait()





        self.wait()
        self.embed()