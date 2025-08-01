from manimlib import *
import numpy as np
import glob
import torch

sys.path.append('_2025/backprop_3')
from geometric_dl_utils import BaarleNet

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'

svg_path='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/to_manim'
graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class LinearPlane(Surface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale=vertical_viz_scale
        super().__init__(
            # u_range=(-12, 12),
            # v_range=(-12, 12),
            u_range=(-0.7, 0.7),
            v_range=(-0.7, 0.7),
            resolution=(64, 64), #Looks nice at 256, but is slow, maybe crank for final
            color='#00FFFF',
            **kwargs
        )
    
    def uv_func(self, u, v):
        # u maps to x1, v maps to x2, compute z = m1*x1 + m2*x2 + b
        x1 = u
        x2 = v
        z = self.vertical_viz_scale*(self.m1 * x1 + self.m2 * x2 + self.b)
        # Transform to axes coordinate system
        return self.axes.c2p(x1, x2, z)

class LinearPlaneWithGrid(Group):
    """A plane with explicit grid lines"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, 
                 grid_lines=20, **kwargs):
        super().__init__()
        
        # Create the main surface
        plane = LinearPlane(axes, m1, m2, b, vertical_viz_scale, **kwargs)
        self.add(plane)
        
        # Create grid lines
        u_range = (-0.7, 0.7)
        v_range = (-0.7, 0.7)
        
        # Vertical grid lines (constant u)
        for i in range(grid_lines + 1):
            u = u_range[0] + i * (u_range[1] - u_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                v = v_range[0] + j * (v_range[1] - v_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(WHITE, width=0.5, opacity=0.3)
            self.add(grid_line)
        
        # Horizontal grid lines (constant v)
        for i in range(grid_lines + 1):
            v = v_range[0] + i * (v_range[1] - v_range[0]) / grid_lines
            line_points = []
            for j in range(21):  # 21 points along the line
                u = u_range[0] + j * (u_range[1] - u_range[0]) / 20
                x1, x2 = u, v
                z = vertical_viz_scale * (m1 * x1 + m2 * x2 + b)
                line_points.append(axes.c2p(x1, x2, z))
            
            grid_line = VMobject()
            grid_line.set_points_as_corners(line_points)
            grid_line.set_stroke(WHITE, width=0.5, opacity=0.3)
            self.add(grid_line)


#Ok pretty sure that network and plane need to be separte and brought together in editing. 
class p18a(InteractiveScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.move_to(ORIGIN)
        map_img.scale(0.25)

        net=VGroup()
        for p in sorted(glob.glob(svg_path+'/p18_21_to_manim*.svg')):
            net.add(SVGMobject(p)[1:])  

        layer_labels=net[4]
        input_circles=net[6]
        neuron_11_shape=net[7]
        neuron_12_shape=net[8]
        neuron_21_shape=net[9]
        neuron_22_shape=net[10]

        #Alright let me get the variables in place here, then I can start thinking through animations. 
        x1=Tex('x_1', font_size=10).set_color(CHILL_BROWN)
        x1.move_to([0,0.115,0])
        x2=Tex('x_2', font_size=10).set_color(CHILL_BROWN)
        x2.move_to([0,-0.21,0])

        #First layer
        # m11_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m11_1=Tex(r'm^{(1)}_{11}', font_size=6).set_color(CYAN)
        m11_1.move_to([0.27, 0.13, 0])
        # m12_1=Tex(r'\boldsymbol{m^{(1)}_{11}}', font_size=6).set_color(CYAN)
        m12_1=Tex(r'm^{(1)}_{12}', font_size=6).set_color(CYAN)
        m12_1.move_to([0.31, 0.031, 0])

        m21_1=Tex(r'm^{(1)}_{21}', font_size=6).set_color(YELLOW)
        m21_1.move_to([0.33, -0.08, 0])

        m22_1=Tex(r'm^{(1)}_{22}', font_size=6).set_color(YELLOW)
        m22_1.move_to([0.27, -0.20, 0])

        b1_1=Tex(r'+b^{(1)}_{1}', font_size=7).set_color(CYAN)
        b1_1.move_to([0.55, 0.13, 0])

        b2_1=Tex(r'+b^{(1)}_{2}', font_size=7).set_color(YELLOW)
        b2_1.move_to([0.55, -0.20, 0])

        #Second layer
        horizontal_offset=0.55
        m11_2=Tex(r'm^{(2)}_{11}', font_size=6).set_color(CYAN)
        m11_2.move_to([0.27+horizontal_offset, 0.13, 0])

        m12_2=Tex(r'm^{(2)}_{12}', font_size=6).set_color(CYAN)
        m12_2.move_to([0.315+horizontal_offset, 0.031, 0])

        m21_2=Tex(r'm^{(2)}_{21}', font_size=6).set_color(YELLOW)
        m21_2.move_to([0.33+horizontal_offset, -0.08, 0])

        m22_2=Tex(r'm^{(2)}_{22}', font_size=6).set_color(YELLOW)
        m22_2.move_to([0.275+horizontal_offset, -0.20, 0])

        b1_2=Tex(r'+b^{(2)}_{1}', font_size=7).set_color(CYAN)
        b1_2.move_to([0.55+horizontal_offset, 0.13, 0])

        b2_2=Tex(r'+b^{(2)}_{2}', font_size=7).set_color(YELLOW)
        b2_2.move_to([0.55+horizontal_offset, -0.20, 0])

        self.add(input_circles, neuron_11_shape, neuron_12_shape, neuron_21_shape, neuron_22_shape, layer_labels)


        self.frame.reorient(0, 0, 0, (0.48, -0.05, 0.0), 1.99)
        self.add(x1, x2, m11_1, m12_1, m21_1, m22_1, b1_1, b2_1)
        self.add(m11_2, m12_2, m21_2, m22_2, b1_2, b2_2)

        # Ok that looks pretty good -> bring in a plane on an axes as a little sketch now maybe? 
        # And then I can really start bringing stuff together??



        self.wait(20)
        self.embed()



class p18b(InteractiveScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        axes_1 = ThreeDAxes(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            width=4,
            height=4,
            depth=3,
            axis_config={
                "color": FRESH_TAN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":4,
                "tip_config": {"width":0.08, "length":0.08}
                }
        )

        w=model.model[0].weight.detach().numpy()
        b=model.model[0].bias.detach().numpy()
        vertical_viz_scale=0.3
        plane_1=LinearPlaneWithGrid(axes_1, w[0,1], w[0,0], b[0], vertical_viz_scale=vertical_viz_scale, grid_lines=12)
        plane_1.set_opacity(0.5)
        plane_1.set_color(CYAN)

        axis_and_plane_11=Group(plane_1, axes_1)

        x1=Tex('x_1', font_size=48).set_color(FRESH_TAN)
       
        x1.rotate(90*DEGREES, [1,0,0])
        x1.next_to(axes_1[0].get_end(), buff=0.15)
        x2=Tex('x_2', font_size=48).set_color(FRESH_TAN)
        x2.rotate(90*DEGREES, [1,0,0])
        x2.rotate(90*DEGREES, [0,0,1])
        x2.next_to(axes_1[1].get_end(), buff=0.15, direction=np.array([0,1,0]))

        h1=Tex('h_1^{(1)}', font_size=36).set_color(CYAN)
        h1.rotate(90*DEGREES, [1,0,0])
        h1.rotate(45*DEGREES, [0,0,1])
        h1.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))



        self.frame.reorient(44, 64, 0, (0.37, 0.13, -0.38), 6.00)
        self.add(axis_and_plane_11)
        self.add(x1, x2, h1)






        self.wait()

        self.wait(20)
        self.embed()



