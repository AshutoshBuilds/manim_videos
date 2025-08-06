from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes
from manimlib import *
from tqdm import tqdm
from order_matching_tools import reorder_polygons_optimal, reorder_polygons_greedy

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
colors = [BLUE, GREY, GREEN, TEAL, PURPLE, PINK, TEAL, YELLOW, FRESH_TAN, CHILL_BLUE, CHILL_GREEN, YELLOW_FADE]
# colors = [GREY, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]


class p64(InteractiveScene):
    '''Ok so this is the transition of my 3d shape for the 2d view -> try blending in premiere!'''
    def construct(self):

        model = BaarleNet([32, 32, 32, 32])
        viz_scales=[0.06, 0.06, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.15]
        num_neurons=[32, 32, 32, 32, 32, 32, 32, 32, 2]
        vertical_spacing=1.0


        data_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/hackin/training_caches/32_32_32_32_1.pkl'
        with open(data_path, 'rb') as file:
            training_cache = pickle.load(file) #Training cache

        prev_layer_1_polygons=None
        prev_layer_2_polygons=None
        prev_layer_3_polygons=None
        prev_layer_4_polygons=None

        self.frame.reorient(0, 0, 0, (3.94, 0.38, 0.0), 2.04)
        # for train_step in tqdm(range(len(training_cache['weights_history']))): #All steps
        # for train_step in tqdm(np.arange(0, len(training_cache['weights_history']), 100)):
        # for train_step in tqdm(np.arange(0,  1000, 1)):
        # for train_step in tqdm(np.arange(0, 600, 100)):
        train_step=0

        if 'layer_1_polygons_flat' in locals():
        	self.remove(layer_1_polygons_flat, layer_2_polygons_flat, layer_3_polygons_flat, layer_3_polygons_flat, final_map_group)


        w1=training_cache['weights_history'][train_step]['model.0.weight'].numpy()
        b1=training_cache['weights_history'][train_step]['model.0.bias'].numpy()
        w2=training_cache['weights_history'][train_step]['model.2.weight'].numpy()
        b2=training_cache['weights_history'][train_step]['model.2.bias'].numpy()
        w3=training_cache['weights_history'][train_step]['model.4.weight'].numpy()
        b3=training_cache['weights_history'][train_step]['model.4.bias'].numpy()
        w4=training_cache['weights_history'][train_step]['model.6.weight'].numpy()
        b4=training_cache['weights_history'][train_step]['model.6.bias'].numpy()

        with torch.no_grad():
            model.model[0].weight.copy_(torch.from_numpy(w1))
            model.model[0].bias.copy_(torch.from_numpy(b1))
            model.model[2].weight.copy_(torch.from_numpy(w2))
            model.model[2].bias.copy_(torch.from_numpy(b2))
            model.model[4].weight.copy_(torch.from_numpy(w3))
            model.model[4].bias.copy_(torch.from_numpy(b3))
            model.model[6].weight.copy_(torch.from_numpy(w4))
            model.model[6].bias.copy_(torch.from_numpy(b4))



        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.6, extent=1)
        #For the interesection to make sense, these scales need to match - either need to manual overide or chnage method above
        final_layer_viz=scale=1.4*min(adaptive_viz_scales[-1]) #little manual ramp here
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
            
            #Compute new tiling - general method with merging - should be more accurate but slow - buggy?
            # polygons[str(layer_id)+'.new_tiling']=recompute_tiling_general(polygons[str(layer_id)+'.split_polygons_merged'])
            
            #Less general method - less accurate, faster, maybe less buggy
            # polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling_nested']=recompute_tiling_polygonize(polygons[str(layer_id)+'.split_polygons_nested_clipped'])
            polygons[str(layer_id)+'.new_tiling']=[item for sublist in polygons[str(layer_id)+'.new_tiling_nested'] for item in sublist]


            print('Retiled plane into ', str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons.')
            #Optional filtering step
            #polygons[str(layer_id)+'.new_tiling'] = filter_small_polygons(polygons[str(layer_id)+'.new_tiling'], min_area=1e-5)
            #print(str(len(polygons[str(layer_id)+'.new_tiling'])), ' polygons remaining after filtering out small polygons')


        #Last linear layer & output
        polygons[str(layer_id+1)+'.linear_out']=process_with_layers(model.model, polygons[str(layer_id)+'.new_tiling'])
        intersection_lines, new_2d_tiling, upper_polytope, indicator = intersect_polytopes(*polygons[str(layer_id+1)+'.linear_out'])
        my_indicator, my_top_polygons = compute_top_polytope(model, new_2d_tiling)


        if prev_layer_1_polygons is not None: 
            prev_layer_1_polygons=reorder_polygons_optimal(prev_layer_1_polygons, polygons['0.new_tiling'])
        else:
            prev_layer_1_polygons=polygons['0.new_tiling']
        layer_1_polygons_flat=manim_polygons_from_np_list(prev_layer_1_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)

        if prev_layer_2_polygons is not None: 
            prev_layer_2_polygons=reorder_polygons_optimal(prev_layer_2_polygons, polygons['1.new_tiling'])
        else:
            prev_layer_2_polygons=polygons['1.new_tiling']         
        layer_2_polygons_flat=manim_polygons_from_np_list(prev_layer_2_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)

        if prev_layer_3_polygons is not None: 
            prev_layer_3_polygons=reorder_polygons_optimal(prev_layer_3_polygons, polygons['2.new_tiling'])
        else:
            prev_layer_3_polygons=polygons['2.new_tiling']
        layer_3_polygons_flat=manim_polygons_from_np_list(prev_layer_3_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)

        if prev_layer_4_polygons is not None: 
            prev_layer_4_polygons=reorder_polygons_optimal(prev_layer_4_polygons, polygons['3.new_tiling'])
        else:
            prev_layer_4_polygons=polygons['2.new_tiling']
        layer_4_polygons_flat=manim_polygons_from_np_list(prev_layer_4_polygons, colors=colors, viz_scale=viz_scales[2], opacity=0.6)






        self.wait(20)
        self.embed()






