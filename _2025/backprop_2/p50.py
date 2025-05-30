from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial
import numpy as np
import torch

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim'

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
            u_range=(-5, 5),
            v_range=(-5, 5),
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



class p50_sketch(InteractiveScene):
    def construct(self):

        baarle_map=ImageMobject(svg_path +'/map_exports_square.00_00_21_24.Still001.png')
        # baarle_map.rotate(90*DEGREES, [1,0,0])

        baarle_map.scale(0.25)
        # baarle_map.move_to([0.96,0,0])

        self.add(baarle_map)
        # self.frame.reorient(0, 90, 0, (0.02, -0.01, 0.0), 4.74)
        self.wait()

        # Ok I think pan down and planes and axis fade in, right?



        axes_1 = ThreeDAxes(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-3.5, 3.5, 1],
            width=1,
            height=1,
            depth=1,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        plane_1=LinearPlane(axes_1, 0.5, 1.2, 4, vertical_viz_scale=0.2)
        plane_1.set_opacity(0.4)
        plane_1.set_color('#00FFFF')

        plane_2=LinearPlane(axes_1, -1, 0.2, 4, vertical_viz_scale=0.2)
        plane_2.set_opacity(0.4)
        plane_2.set_color(YELLOW)

        self.add(axes_1, plane_1, plane_2)

        self.frame.reorient(-29, 53, 0, (0.04, 0.06, 0.09), 2.05)

        #ok static looks good, now I want to pan around while smootly changing the LinearPlane's parameters. 
        # start_position reorient(-33, 58, 0, (-0.1, 0.01, 0.05), 1.83)
        # start_position reorient(33, 59, 0, (-0.1, 0.01, 0.05), 1.83)





        self.wait()
        self.embed()



























