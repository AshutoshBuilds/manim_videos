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
        

        net=VGroup()
        for p in sorted(glob.glob(svg_path+'/p18_21_to_manim*.svg')):
            net.add(SVGMobject(p)[1:])  

        map_frame=net[1]
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

        # self.add(input_circles, neuron_11_shape, neuron_12_shape, neuron_21_shape, neuron_22_shape, layer_labels)
        inputs_group=Group(input_circles, x1, x2)
        neuron_11_group=Group(neuron_11_shape, m11_1, m12_1, b1_1)
        neuron_12_group=Group(neuron_12_shape, m21_1, m22_1, b2_1)
        neuron_21_group=Group(neuron_21_shape, m11_2, m12_2, b1_2)
        neuron_22_group=Group(neuron_22_shape, m21_2, m22_2, b2_2)

        # self.frame.reorient(0, 0, 0, (0.48, -0.05, 0.0), 1.99)
        # Ok so I'll so smooth handoff from illustrator to this - with the network towards the bottom of the screen
        # I'll add in the 3d plane from 18b above in editing
        # self.add(x1, x2, m11_1, m12_1, m21_1, m22_1, b1_1, b2_1)
        # self.add(m11_2, m12_2, m21_2, m22_2, b1_2, b2_2)
        self.frame.reorient(0, 0, 0, (0.52, 0.38, 0.0), 2.00)
        self.add(layer_labels)
        self.add(inputs_group, neuron_11_group, neuron_12_group, neuron_21_group, neuron_22_group)
        self.wait()

        #Ok now lower opacity on everything except the first neuron!
        self.play(
                  # inputs_group.animate.set_opacity(0.3), 
                  neuron_12_group.animate.set_opacity(0.3),
                  neuron_21_group.animate.set_opacity(0.3),
                  neuron_22_group.animate.set_opacity(0.3),
                  layer_labels.animate.set_opacity(0.3), 
                    run_time=2)
        self.wait()

        #Now build out equation!
        eq1=Tex(r'h_{1}^{(1)}=m_{11}^{(1)}x_{1}+m_{12}^{(1)}x_{2}+b_1^{(1)}', font_size=8).set_color(FRESH_TAN)
        eq1.move_to([1.2, 0.8, 0])
        eq1[6:12].set_color(CYAN) #m11_1
        eq1[15:21].set_color(CYAN) #m12_1
        eq1[24:].set_color(CYAN) #b_1

        self.wait()
        self.play(ReplacementTransform(x1.copy(), eq1[12:14]),
                  ReplacementTransform(x2.copy(), eq1[21:23]),
                  run_time=3 
                 )

        self.wait()
        self.play(ReplacementTransform(m11_1.copy(), eq1[6:12]),
                  ReplacementTransform(m12_1.copy(), eq1[15:21]),
                  run_time=3 
                 )
        self.wait()
        self.add(eq1[14])

        self.play(ReplacementTransform(b1_1.copy(), eq1[23:]),run_time=3)
        self.wait()

        # Ok sick, so we've got a basic equation, and the plane in frame, 
        # now it's time to add the map, right? Ok I think let's just scale it down and put it in lower left corner. 
        #Let's try it
        map_img.scale(0.22)
        map_img.move_to([-0.65, -0.04, 0])
        map_frame.scale(0.764)
    
        map_frame.move_to(map_img)
        map_frame.shift([0, 0.015, 0])
        self.play(FadeIn(map_img), FadeIn(map_frame))
        self.wait()

        #ok now we need our point on that map!
        map_coords=Tex(r'(0.6, 0.4)', font_size=11).set_color('#FF00FF')
        map_pt=Dot(ORIGIN, radius=0.015).set_color('#FF00FF')
        map_pt.move_to([-0.39, 0.12, 0])
        map_coords.move_to([-0.4, 0.2, 0])
        self.add(map_pt, map_coords)

        #Ok pieces together next equation, then move things over
        #2.51*0.6+(-1.02)*0.4-1.24
        eq2=Tex(r'h_{1}^{(1)}=(2.51)(0.6)+(-1.02)(0.4)+(-1.22)', font_size=6).set_color(FRESH_TAN)
        eq2.move_to([1.25, 0.65, 0])
        eq2[6:12].set_color(CYAN) #m11_1
        eq2[12:17].set_color('#FF00FF')
        eq2[18:25].set_color(CYAN) #m12_1
        eq2[25:30].set_color('#FF00FF')
        eq2[31:].set_color(CYAN) #b_1

        eq3=Tex(r'=-0.14', font_size=8).set_color(CYAN)
        eq3.move_to([0.95, 0.53, 0])
        self.wait()

        eq1_copy=eq1.copy()
        self.play(ReplacementTransform(map_coords[1:4].copy(), eq2[13:16]), run_time=3)
        self.play(Transform(eq1_copy[6:12], eq2[6:12]), run_time=1.5)
        self.add(eq2[5], eq2[12:17])
        self.wait()

        self.play(ReplacementTransform(map_coords[5:8].copy(), eq2[26:29]), run_time=3)
        self.play(ReplacementTransform(eq1_copy[15:21], eq2[18:25]), run_time=1.5)
        self.add(eq2[17], eq2[25:30])
        self.wait()

        self.play(ReplacementTransform(eq1_copy[24:], eq2[31:]), run_time=1.5)
        self.add(eq2[30])     
        self.wait()   

        self.play(Write(eq3))
        self.wait()

        # Ok so at this point, I need to add the hieght of this to my plane p18b
        # So I think on p19 I bring back up the opacity of the last two neurons
        # And then do the collapsing trick, I think right?

        self.play(
                  neuron_21_group.animate.set_opacity(1.0),
                  neuron_22_group.animate.set_opacity(1.0),
                  run_time=2)
        self.wait()

        # Ok ok ok ok ok ok ok now how do I animate this collapsing thing?
        # Fade out middle layer, and then bring inputs over? That might kinda work?
        # Ok taking a break then will try that.

        self.remove(inputs_group); self.add(inputs_group)
        self.wait(0)
        self.play(FadeOut(neuron_12_group), 
                  FadeOut(neuron_11_group), 
                  FadeOut(layer_labels),
                  inputs_group.animate.move_to([0.55,-0.03,0]),
                  run_time=3)
        self.wait()

        #Ok now we'll fade things out here, I think uncollapse networks so I can show all symbols. 

        self.wait()
        self.remove(eq1, eq1_copy, eq2, eq3) #Can't seem to get a nice fade out on these - just remove I think!
        self.play(
                  # FadeIn(neuron_12_group), 
                  FadeIn(neuron_11_group),
                  neuron_12_group.animate.set_opacity(1.0),
                  FadeOut(map_img),
                  FadeOut(map_frame),
                  # eq3.animate.set_opacity(0.0),
                  # eq2.animate.set_opacity(0.0),
                  # eq1_copy.animate.set_opacity(0.0),
                  # eq1.animate.set_opacity(0.0),
                  FadeOut(map_pt),
                  FadeOut(map_coords),
                  inputs_group.animate.move_to([-7.40739175e-03, -3.67778116e-02, 0]),
                  run_time=3)
        self.wait()
        
        #Hmm that might actually be a nice clean breakpoint a next scene?

        self.wait(20)
        self.embed()

class p20_21(InteractiveScene):
    def construct(self):

        model_path='_2025/backprop_3/models/2_1.pth'
        model = BaarleNet([2])
        model.load_state_dict(torch.load(model_path))


        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.move_to(ORIGIN)
        

        net=VGroup()
        for p in sorted(glob.glob(svg_path+'/p18_21_to_manim*.svg')):
            net.add(SVGMobject(p)[1:])  

        map_frame=net[1]
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

        # self.add(input_circles, neuron_11_shape, neuron_12_shape, neuron_21_shape, neuron_22_shape, layer_labels)
        inputs_group=Group(input_circles, x1, x2)
        neuron_11_group=Group(neuron_11_shape, m11_1, m12_1, b1_1)
        neuron_12_group=Group(neuron_12_shape, m21_1, m22_1, b2_1)
        neuron_21_group=Group(neuron_21_shape, m11_2, m12_2, b1_2)
        neuron_22_group=Group(neuron_22_shape, m21_2, m22_2, b2_2)

        # self.frame.reorient(0, 0, 0, (0.48, -0.05, 0.0), 1.99)
        # Ok so I'll so smooth handoff from illustrator to this - with the network towards the bottom of the screen
        # I'll add in the 3d plane from 18b above in editing
        # self.add(x1, x2, m11_1, m12_1, m21_1, m22_1, b1_1, b2_1)
        # self.add(m11_2, m12_2, m21_2, m22_2, b1_2, b2_2)
        self.frame.reorient(0, 0, 0, (0.52, 0.38, 0.0), 2.00)
        # self.add(layer_labels)
        self.add(inputs_group, neuron_11_group, neuron_12_group, neuron_21_group, neuron_22_group)
        self.wait()

        h1_2=Tex(r'h^{(2)}_{1}', font_size=8).set_color(CHILL_BROWN)
        h1_2.move_to([1.43, 0.13, 0])
        h2_2=Tex(r'h^{(2)}_{2}', font_size=8).set_color(CHILL_BROWN)
        h2_2.move_to([1.43, -0.2, 0])
        # self.add(h1_2, h2_2)

        self.wait()
        self.play(FadeIn(h1_2), FadeIn(h2_2), 
                  m11_2.animate.set_color('#FF00FF'), m12_2.animate.set_color('#FF00FF'), b1_2.animate.set_color('#FF00FF'),
                  m21_2.animate.set_color(CHILL_GREEN), m22_2.animate.set_color(CHILL_GREEN), b2_2.animate.set_color(CHILL_GREEN))
        self.wait()



        # Ok now let's build these equqations out here!
        # Hmm do I want to consider a bit of a color change here? 
        # like make all the neurons match?? 

        eq1 = Tex(r"h_{1}^{(1)} = m_{11}^{(1)} x_{1} + m_{12}^{(1)} x_{2} + b_{1}^{(1)}", font_size=10).set_color(FRESH_TAN)
        eq2 = Tex(r"h_{2}^{(1)} = m_{21}^{(1)} x_{1} + m_{22}^{(1)} x_{2} + b_{2}^{(1)}", font_size=10).set_color(FRESH_TAN)
        eq3 = Tex(r"h_{1}^{(2)} = m_{11}^{(2)} h_{1}^{(1)} + m_{12}^{(2)} h_{2}^{(1)} + b_{1}^{(2)}", font_size=10).set_color(FRESH_TAN)
        # eq3[12:17].set_color(CYAN)
        # eq3[24:29].set_color(YELLOW)

        eq1.move_to([0.55, 1.1, 0])
        eq2.move_to([0.55, 0.9, 0])
        eq3.move_to([0.55, 0.7, 0])
        
        eq3[12:17].set_color(CYAN)
        eq3[24:29].set_color(YELLOW)

        eq1[:5].set_color(CYAN)
        eq1[6:12].set_color(CYAN) #m11_1
        eq1[15:21].set_color(CYAN) #m12_1
        eq1[24:].set_color(CYAN) #b_1

        eq2[:5].set_color(YELLOW)
        eq2[6:12].set_color(YELLOW) #m11_1
        eq2[15:21].set_color(YELLOW) #m12_1
        eq2[24:].set_color(YELLOW) #b_1

        # eq3[:5].set_color('#FF00FF')
        eq3[6:12].set_color('#FF00FF') #m11_1
        eq3[18:24].set_color('#FF00FF') #m12_1
        eq3[29:].set_color('#FF00FF') #b_1

        # self.add(eq1, eq2, eq3)
        self.wait()

        #Ok now build equations from copies of values from neuron drawing!
        self.play(ReplacementTransform(m11_1.copy(), eq1[6:12]), 
                  ReplacementTransform(m12_1.copy(), eq1[15:21]), 
                  ReplacementTransform(b1_1.copy(), eq1[24:]), 
                  run_time=3)
        self.add(eq1)
        self.play(ReplacementTransform(m21_1.copy(), eq2[6:12]), 
                  ReplacementTransform(m22_1.copy(), eq2[15:21]), 
                  ReplacementTransform(b2_1.copy(), eq2[24:]), 
                  run_time=3)
        self.add(eq2)
        self.wait()
        self.play(ReplacementTransform(m11_2.copy(), eq3[6:12]), 
                  ReplacementTransform(m12_2.copy(), eq3[18:24]), 
                  ReplacementTransform(b1_2.copy(), eq3[29:]), 
                  run_time=3)
        self.add(eq3)
        self.wait()


        # neuron_11_group.set_color(PURPLE)

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

        axis_and_plane_11=Group(axes_1, plane_1)

        x1=Tex('x_1', font_size=52).set_color(FRESH_TAN)
        x1.rotate(90*DEGREES, [1,0,0])
        x1.next_to(axes_1[0].get_end(), buff=0.15)
        x2=Tex('x_2', font_size=52).set_color(FRESH_TAN)
        x2.rotate(90*DEGREES, [1,0,0])
        x2.rotate(90*DEGREES, [0,0,1])
        x2.next_to(axes_1[1].get_end(), buff=0.15, direction=np.array([0,1,0]))

        h1=Tex('h_1^{(1)}', font_size=42).set_color(CYAN)
        h1.rotate(90*DEGREES, [1,0,0])
        h1.rotate(45*DEGREES, [0,0,1])
        h1.next_to(axes_1[2].get_end(), buff=0.15, direction=np.array([0,0,1]))


        self.frame.reorient(44, 73, 0, (0.14, 0.04, -0.1), 6.00)
        self.add(axis_and_plane_11)
        self.add(x1, x2, h1)

        #Add point, connecting lines and maybe a nice little twist zoom in animation
        dot_point = axes_1.c2p(0.6, 0.4, -0.25)
        magenta_dot = Sphere(radius=0.1, color='#FF00FF')
        magenta_dot.move_to(dot_point)

        line1_start = axes_1.c2p(0.6, 0, 0)
        line1_end = axes_1.c2p(0.6, 0.4, 0)
        dashed_line1 = DashedLine(line1_start, line1_end, color='#FF00FF', stroke_width=8)
        dashed_line1.rotate(90*DEGREES, [0,1,0])
        
        line2_start = axes_1.c2p(0.0, 0.4, 0)
        line2_end = axes_1.c2p(0.6, 0.4, 0)
        dashed_line2 = DashedLine(line2_start, line2_end, color='#FF00FF', stroke_width=8)
        dashed_line2.rotate(90*DEGREES, [1,0,0])

        line3_start = axes_1.c2p(0.6, 0.4, 0)
        line3_end = axes_1.c2p(0.6, 0.4, -0.25)
        dashed_line3 = DashedLine(line3_start, line3_end, color='#FF00FF', stroke_width=8)
        # dashed_line2.rotate(90*DEGREES, [1,0,0])

        # self.add(magenta_dot, dashed_line1, dashed_line2, dashed_line3)
        self.wait()
        self.play(self.frame.animate.reorient(69, 68, 0, (0.14, 0.09, -0.06), 5.11), 
                  FadeIn(dashed_line1), FadeIn(dashed_line2), FadeIn(dashed_line3), FadeIn(magenta_dot),
                 run_time=4.0)
        self.wait()





        self.wait()

        self.wait(20)
        self.embed()



