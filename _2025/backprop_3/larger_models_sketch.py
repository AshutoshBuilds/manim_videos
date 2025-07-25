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
        
        #2x2
        # model_path='_2025/backprop_3/models/2_2_1.pth'
        # model = BaarleNet([2,2])
        # model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.25, 0.25, 0.3, 0.3, 0.15]
        # num_neurons=[2, 2, 2, 2, 2]

        #3x3
        model_path='_2025/backprop_3/models/3_3_1.pth'
        model = BaarleNet([3,3])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        num_neurons=[3, 3, 3, 3, 2]

        #8x8
        model_path='_2025/backprop_3/models/8_8_1.pth'
        model = BaarleNet([8,8])
        model.load_state_dict(torch.load(model_path))
        # viz_scales=[0.1, 0.1, 0.05, 0.05, 0.15]
        num_neurons=[8,8, 8, 8, 2]

        vertical_spacing=1.5
        horizontal_spacing=3
        colors = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]

        # Hmm ok I think I need to compute a different viz scale for each surface probably 
        # I don't want to totally normalize the heights though, ya know?
        # Maybe there's a discrete set of possible viz scales: 
        adaptive_viz_scales = compute_adaptive_viz_scales(model, max_surface_height=0.75, extent=1)


        surfaces=[]
        surface_funcs=[]
        surface_funcs_no_viz_scale=[]
        for layer_idx in range(len(model.model)):
            s=Group()
            surface_funcs.append([])
            surface_funcs_no_viz_scale.append([])
            for neuron_idx in range(num_neurons[layer_idx]):
                surface_func=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=adaptive_viz_scales[layer_idx][neuron_idx])
                surface_func_no_scaling=partial(surface_func_from_model, model=model, layer_idx=layer_idx, neuron_idx=neuron_idx, viz_scale=1.0) #adaptive_viz_scales[layer_idx][neuron_idx])
                bent_surface = ParametricSurface(surface_func, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
                ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
                ts.set_shading(0,0,0).set_opacity(0.75)
                s.add(ts)
                surface_funcs[-1].append(surface_func)
                surface_funcs_no_viz_scale[-1].append(surface_func_no_scaling)
            surfaces.append(s)

        for layer_idx, sl in enumerate(surfaces):
            for neuron_idx, s in enumerate(sl):
                s.shift([horizontal_spacing*layer_idx-6, 0, vertical_spacing*neuron_idx])
                self.add(s)


        self.frame.reorient(0, 54, 0, (1.41, 1.82, 4.15), 15.71)


        #Optional but kinda nice RuLu intersection planes
        layer_idx=0
        relu_intersections_planes_1=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_1)


        #Layer 1 polygons - use surface function to find heights
        layer_idx=1
        layer_1_polygons=get_polygon_corners_layer_1(model)

        #I think i need to use unscaled surface function here so Relus hit at the right spot
        layer_1_polygons_3d=get_3d_polygons_layer_1(layer_1_polygons, surface_funcs_no_viz_scale, num_neurons=num_neurons[layer_idx], layer_idx=1)
        
        #Ok now I need to rescale. 
        scaled_layer_1_polygons_3d=apply_viz_scale_to_3d_polygons(layer_1_polygons_3d, adaptive_viz_scales[layer_idx])

        polygons_vgroup=viz_3d_polygons(scaled_layer_1_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup)
        self.wait()


        #Shadow of layer 1 polygons, basically the regions available to our second layer
        layer_2_polygons=carve_plane_with_relu_joints([o['relu_line'] for o in layer_1_polygons])


        #2d shadow of these regions
        output_poygons_2d=viz_carved_regions_flat(layer_2_polygons, horizontal_spacing, layer_idx, colors=None)
        self.add(output_poygons_2d)


        #Layer 2 linear
        layer_idx=2
        layer_2_polygons_3d=get_3d_polygons(layer_2_polygons, num_neurons[layer_idx], surface_funcs_no_viz_scale, layer_idx)
        scaled_layer_2_polygons_3d=apply_viz_scale_to_3d_polygons(layer_2_polygons_3d, adaptive_viz_scales[layer_idx])


        polygons_vgroup_2=viz_3d_polygons(scaled_layer_2_polygons_3d, layer_idx, colors=None)
        self.add(polygons_vgroup_2)


   
        relu_intersections_planes_2=get_relu_intersection_planes(num_neurons[layer_idx], layer_idx, neuron_idx, horizontal_spacing, vertical_spacing)
        self.add(relu_intersections_planes_2)


        #Layer 2 Relu
        layer_idx=3
        all_polygons, merged_zero_polygons, unmerged_polygons = split_polygons_with_relu(layer_2_polygons_3d)


        #Ok a little clunky to do a post merge like this, but I think this gives some good flexiblity!
        all_polygons_after_merging=copy.deepcopy(merged_zero_polygons)
        for i, o in enumerate(unmerged_polygons):
            all_polygons_after_merging[i].extend(o)


        all_polygons_after_merging_scaled=apply_viz_scale_to_3d_polygons(all_polygons_after_merging, adaptive_viz_scales[layer_idx])

        # layer_2_polygons_split_vgroup=viz_3d_polygons(all_polygons_after_merging, layer_idx, colors=None)
        # self.add(layer_2_polygons_split_vgroup)


        layer_2_polygons_split_vgroup=VGroup()
        layer_2_colors = [GREY, RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
        for neuron_idx, polygons in enumerate(all_polygons_after_merging_scaled):
            for j, p in enumerate(polygons):
                poly_3d = Polygon(*p,
                                 fill_color=layer_2_colors[j%len(layer_2_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_2_colors[j%len(layer_2_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_2_polygons_split_vgroup.add(poly_3d)
        self.add(layer_2_polygons_split_vgroup)


        #Making progress! I think I need to mess with the scaling a little though?







        # Ok this is looking pretty good. 
        # So I need to do another merge step -> 
        # At layer 1 i was able to get away with carve_plane_with_relu_joints
        # Same principle here, but obviously a bit more complex
        # Now my intuition, and the animation that comes to mind, is dropping down all the lines
        # from all 3 layers and then recarcing based on the number of intersections
        # between all these lines - is that right? Line combination happens in 2d basically?
        # Ok i'm like 90% sure that's right, if it's not I think it will shake out in the viz
        # Let's assume it's right and see what happens here exaclty. 

        #Drop last coords. 
        all_polygons_after_merging_2d=[]
        for p in all_polygons_after_merging:
            pd2=[o[:,:2] for o in p]
            all_polygons_after_merging_2d.append(pd2)

        # layer3_regions_2d = compute_layer3_regions(all_polygons_after_merging)
        layer3_regions_2d = find_polygon_intersections(all_polygons_after_merging_2d)

        #Let's do a quick 2d viz to see how things are looking here
        output_poygons_2d=VGroup()
        layer_idx=4
        neuron_idx=-1
        layer_3_colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]
        for j, polygon in enumerate(layer3_regions_2d):
                polygon = Polygon(*np.hstack((polygon, np.zeros((polygon.shape[0],1)))),
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                polygon.set_opacity(0.3)
                polygon.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                output_poygons_2d.add(polygon)
        self.add(output_poygons_2d)

        # Ok making progress here - now last step and then I want to think about exponential growth here
        # Last step will be sending these to 3d! Which should be maybe not terrible?

        layer_3_polygons_3d=[]
        for neuron_idx in range(num_neurons[layer_idx]):
            layer_3_polygons_3d.append([])
            for polygon_2d in layer3_regions_2d:
                a=[]
                for pt_idx in range(len(polygon_2d)):
                    a.append(surface_funcs[layer_idx][neuron_idx](*polygon_2d[pt_idx])) #Might be a batch way to do this
                a=np.array(a)
                layer_3_polygons_3d[-1].append(a)           


        layer_3_polygons_vgroup=VGroup()
        for neuron_idx, polygons in enumerate(layer_3_polygons_3d):
            for j, p in enumerate(polygons):
                poly_3d = Polygon(*p,
                                 fill_color=layer_3_colors[j%len(layer_3_colors)],
                                 fill_opacity=0.7,
                                 stroke_color=layer_3_colors[j%len(layer_3_colors)],
                                 stroke_width=2)
                poly_3d.set_opacity(0.3)
                poly_3d.shift([3*layer_idx-6, 0, 1.5*neuron_idx])
                layer_3_polygons_vgroup.add(poly_3d)
        self.add(layer_3_polygons_vgroup)






        self.wait()

        

        self.wait()





        self.wait()
        self.embed()






