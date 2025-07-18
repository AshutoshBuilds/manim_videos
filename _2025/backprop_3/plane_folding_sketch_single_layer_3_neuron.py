from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from plane_folding_utils import *

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class plane_folding_sketch_single_layer_2(InteractiveScene):
    def construct(self):

        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.set_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.set_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(ORIGIN)

        viz_scale_1=0.25
        viz_scale_2=0.1

        #New 3 layer weights!
        w1 = np.array([[-2.00458, 2.24611],
         [-2.56046, -1.21349],
         [-1.94774, 0.716835]], dtype=np.float32)
        b1 = np.array([0.00728259, -1.38003, 1.77056], dtype=np.float32)
        w2 = np.array([[2.46867, 3.78735, -1.90977],
         [-2.55351, -2.95687, 1.74294]], dtype=np.float32)
        b2 = np.array([1.41342, -1.23457], dtype=np.float32)

        # First hidden neuron
        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=viz_scale_1)
        bent_surface_11 = ParametricSurface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts11=TexturedSurface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts11.set_shading(0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.5)
        group_11=Group(ts11, joint_line_11)

        # Second hidden neuron
        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=viz_scale_1)
        bent_surface_12 = ParametricSurface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts12=TexturedSurface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts12.set_shading(0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.5)
        group_12=Group(ts12, joint_line_12)

        # Third hidden neuron (NEW)
        surface_func_13=partial(surface_func_general, w1=w1[2,0], w2=w1[2,1], b=b1[2], viz_scale=viz_scale_1)
        bent_surface_13 = ParametricSurface(surface_func_13, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts13=TexturedSurface(bent_surface_13, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts13.set_shading(0,0,0)
        ts13.set_opacity(0.75)
        joint_points_13 = get_relu_joint(w1[2,0], w1[2,1], b1[2], extent=1)
        joint_line_13=line_from_joint_points_1(joint_points_13).set_opacity(0.5)
        group_13=Group(ts13, joint_line_13)

        self.frame.reorient(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        self.add(group_11, group_12, group_13)
        
        # Position the three hidden layer visualizations
        group_13.shift([-3, 0, 3])    # Top
        group_12.shift([-3, 0, 1.5])  # Middle
        group_11.shift([-3, 0, 0])    # Bottom
        self.wait()

        # First output neuron
        neuron_idx=0
        surface_func_21 = partial(
            surface_func_second_layer_no_relu_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_21 = ParametricSurface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts21 = TexturedSurface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts21.set_shading(0,0,0)
        ts21.set_opacity(0.75)

        bs21_copy=bent_surface_21.copy()
        ts21_copy=ts21.copy()

        # Get all joint points for the 3 neurons
        joint_points_list = [joint_points_11, joint_points_12, joint_points_13]
        polygons = get_polygon_corners_multi(joint_points_list, extent=1)

        ts21.shift([0, 0, 1.5])
        self.add(ts21)
        
        # Create 3D polygon regions for first output neuron
        polygon_3d_objects = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        polygon_3d_objects_copy = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            poly.shift([0, 0, 1.5])
            self.add(poly)
        self.wait()

        # Second output neuron
        neuron_idx=1
        surface_func_22 = partial(
            surface_func_second_layer_no_relu_multi, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )

        bent_surface_22 = ParametricSurface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(64, 64))
        ts22 = TexturedSurface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts22.set_shading(0,0,0)
        ts22.set_opacity(0.75)

        bs22_copy=bent_surface_22.copy()
        ts22_copy=ts22.copy()

        self.add(ts22)

        
        # Create 3D polygon regions for second output neuron
        polygon_3d_objects_2 = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        polygon_3d_objects_2_copy = create_3d_polygon_regions_multi(
            polygons, w1, b1, w2, b2, neuron_idx=neuron_idx, viz_scale=viz_scale_2
        )
        
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            self.add(poly)
        self.wait()


        # Final visualization positioning
        bs21_copy.move_to([3, 0, 0.75])
        ts21_copy.move_to([3, 0, 0.75])
        bs22_copy.move_to([3, 0, 0.75])
        ts22_copy.move_to([3, 0, 0.75])

        map_img.move_to([3, 0, 0.5])
        self.add(map_img)

        # Add the copy polygons for visualization
        for poly in polygon_3d_objects_copy:
            poly.set_opacity(0.3).set_color(BLUE)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        for poly in polygon_3d_objects_2_copy:
            poly.set_opacity(0.3).set_color(YELLOW)
            poly.shift([3, 0, 0.75])
            self.add(poly)

        self.frame.reorient(11, 62, 0, (3.2, 0.86, 0.02), 4.39)

        self.wait()

        ## Ok this is pretty freaking dope. 
        ## What I want now (and would probably be nice for the 2 neuron case)
        ## is to draw in the borders/intserctions between the two shells
        ## 





        self.wait(20)
        self.embed()






        # colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
        # polygon_mobjects = []

        # for i, polygon_points in enumerate(polygons):
        #     if len(polygon_points) >= 3:
        #         # Convert 2D points to 3D for Manim
        #         points_3d = [[p[0], p[1], 0] for p in polygon_points]
                
        #         # Create the polygon
        #         poly = Polygon(*points_3d, 
        #                       fill_color=colors[i % len(colors)], 
        #                       fill_opacity=0.4,
        #                       stroke_color=colors[i % len(colors)],
        #                       stroke_width=2)
                
        #         polygon_mobjects.append(poly)
        #         self.add(poly)

        # self.wait()

        # self.add(polygon_mobjects[0])
        # self.add(polygon_mobjects[1])
        # self.add(polygon_mobjects[2])
        # self.add(polygon_mobjects[3])
        # self.add(polygon_mobjects[4])


