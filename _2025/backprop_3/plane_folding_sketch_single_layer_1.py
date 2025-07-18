from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from plane_folding_utils import *

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class plane_folding_sketch_single_layer_1(InteractiveScene):
    def construct(self):


        w1=np.array([[-0.70856154,  1.809896  ],
                     [-1.7940422 , -0.4643133 ]], dtype=np.float32)
        b1=np.array([-0.47732198, -1.0138882 ], dtype=np.float32)
        w2=np.array([[ 1.5246898,  2.049856 ],
                    [-1.6014509, -1.3020881]], dtype=np.float32)
        b2=np.array([-0.40461758,  0.05192775], dtype=np.float32)

        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=0.42) #Larger for intput layer
        bent_surface_11 = ParametricSurface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts11=TexturedSurface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts11.set_shading(0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.5)
        group_11=Group(ts11, joint_line_11)

        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=0.42) #Larger for intput layer
        bent_surface_12 = ParametricSurface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts12=TexturedSurface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts12.set_shading(0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.5)
        group_12=Group(ts12, joint_line_12)

        self.frame.reorient(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        self.add(group_11, group_12)
        group_12.shift([-3, 0, 1.5])
        group_11.shift([-3, 0, 0])
        self.wait()

        neuron_idx=0
        surface_func_21 = partial(
            surface_func_second_layer_no_relu, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.25
        )

        bent_surface_21 = ParametricSurface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21 = TexturedSurface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts21.set_shading(0,0,0)
        ts21.set_opacity(0.75)
        polygons = get_polygon_corners(joint_points_11, joint_points_12, extent=1)

        ts21.shift([0, 0, 1.5])
        self.add(ts21)
        polygon_3d_objects = create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            poly.shift([0, 0, 1.5])
            self.add(poly)
        self.wait()


        neuron_idx=1
        surface_func_22 = partial(
            surface_func_second_layer_no_relu, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.25
        )

        bent_surface_22 = ParametricSurface(surface_func_22, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21 = TexturedSurface(bent_surface_22, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts21.set_shading(0,0,0)
        ts21.set_opacity(0.75)

        self.add(ts21)
        polygon_3d_objects_2 = create_3d_polygon_regions( polygons, w1, b1, w2, b2,  neuron_idx=neuron_idx, viz_scale=0.25)
        for poly in polygon_3d_objects_2:
            poly.set_opacity(0.3)
            self.add(poly)
        self.wait()


        self.wait()



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




        self.wait(20)
        self.embed()
