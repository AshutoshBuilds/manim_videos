from manimlib import *
import numpy as np

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are

class LinearPlane(Surface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale = vertical_viz_scale
        super().__init__(
            u_range=(-5, 5),
            v_range=(-5, 5),
            resolution=(64, 64),
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

def find_plane_intersection(plane1, plane2):
    """
    Find the intersection line of two planes.
    Each plane is defined as z = m1*x1 + m2*x2 + b
    
    For plane1: z = m1_1*x1 + m2_1*x2 + b1
    For plane2: z = m1_2*x1 + m2_2*x2 + b2
    
    At intersection: m1_1*x1 + m2_1*x2 + b1 = m1_2*x1 + m2_2*x2 + b2
    Rearranging: (m1_1 - m1_2)*x1 + (m2_1 - m2_2)*x2 = b2 - b1
    
    This gives us a line in the x1-x2 plane. We can parameterize it.
    """
    # Coefficients of the intersection equation
    a = plane1.m1 - plane2.m1  # coefficient of x1
    b = plane1.m2 - plane2.m2  # coefficient of x2  
    c = plane2.b - plane1.b    # constant term
    
    # We need to solve: a*x1 + b*x2 = c
    # We'll parameterize this line using parameter t
    
    if abs(a) > abs(b):  # Use x2 as parameter
        # x1 = (c - b*x2) / a, let x2 = t
        def get_point(t):
            x2 = t
            x1 = (c - b * x2) / a if abs(a) > 1e-10 else 0
            # Calculate z using either plane (they should be equal at intersection)
            z = plane1.vertical_viz_scale * (plane1.m1 * x1 + plane1.m2 * x2 + plane1.b)
            return np.array([x1, x2, z])
    else:  # Use x1 as parameter
        # x2 = (c - a*x1) / b, let x1 = t
        def get_point(t):
            x1 = t
            x2 = (c - a * x1) / b if abs(b) > 1e-10 else 0
            # Calculate z using either plane (they should be equal at intersection)
            z = plane1.vertical_viz_scale * (plane1.m1 * x1 + plane1.m2 * x2 + plane1.b)
            return np.array([x1, x2, z])
    
    return get_point

class IntersectionLine(ParametricCurve):
    """A parametric line representing the intersection of two planes"""
    def __init__(self, axes, plane1, plane2, t_range=(-5, 5, 0.1), **kwargs):
        self.axes = axes
        self.plane1 = plane1
        self.plane2 = plane2
        
        # Get the intersection function
        self.intersection_func = find_plane_intersection(plane1, plane2)
        
        super().__init__(
            lambda t: self.axes.c2p(*self.intersection_func(t)),
            t_range=t_range,
            **kwargs
        )

class p17(InteractiveScene):
    def construct(self):
        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.move_to(ORIGIN)
        map_img.scale(0.25)
        
        axes_1 = ThreeDAxes(
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
        plane_1.set_opacity(0.3)
        plane_1.set_color('#00FFFF')
        
        plane_2=LinearPlane(axes_1, -1, 0.2, 4, vertical_viz_scale=0.2)
        plane_2.set_opacity(0.3)
        plane_2.set_color(YELLOW)
        
        # Create the intersection line
        intersection_line = IntersectionLine(
            axes_1, 
            plane_1, 
            plane_2, 
            t_range=(-5, 5, 0.1),
            color=WHITE,
            stroke_width=4
        )
            

        self.frame.reorient(0, 0, 0, (0.01, -0.01, 0.0), 1.29)
        self.add(map_img)

        self.wait()
        self.play(ShowCreation(plane_1), ShowCreation(plane_2), self.frame.animate.reorient(0, 46, 0, (-0.03, 0.02, 0.01), 1.62), run_time=5)
        self.wait()

        self.play(self.frame.animate.reorient(-16, 47, 0, (-0.04, 0.02, 0.0), 1.62))
        self.play(ShowCreation(intersection_line))
        self.wait()
        
        ## Move to overhead view, bring down line, and each color half plane onto z=0
        

        self.wait()



        self.wait(20)
        self.embed()





