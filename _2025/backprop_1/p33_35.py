from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

def param_surface(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = surf[u_idx, v_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])

class P33(InteractiveScene):
    def construct(self):

        # Ok ok ok ok I need ot decide how automated vs manual I'm going to make this scene
        # Thinking about this scene more, I do think the "right triangle overhead view" of the gradient is going to be nice/important.
        #  
        # Ok ok ok ok ok I do think i want some more Meaat on p33-p35 → specifically it’s interesting and non-obvious that we can put 
        # each slope together in that overhead view and it will point us downhill. Worth thinking about that animation a bit I think → 
        # maybe an actually good use case for WelchAxes lol. And i can draw a connection to part 2 → “as we’ll see in part 2 it turns 
        # out that we can very efficiently estimate the slope of the curves without actually computing an points on them → and then 
        # maybe the little arrows on the curves move to the overhead view? or copies of them? Them more i think about this the more 
        # I think it should be manimy? I can draw the projections/cuts as nice blue/yellow lines in the overhead view too. 
        #
        # Alright kinda low on brain power here, but let me at least try to get the pieces together tonight
        # I do think it probably makes sense to at least try to get all 3 graphs on one canvas, so I can like smoothlly move arrows arround and stuff
        # I might need to go compute a bunch of gradients eh?


        surf=1.6*np.load('_2025/backprop_1/p_24_28_losses_4.npy') #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load('_2025/backprop_1/p_24_28_losses_4xy.npy')


        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=VGroup(x_axis_1, y_axis_1)
        axes_1.move_to([-3.5, 1.5, 0])
        self.add(axes_1)

        x_axis_1.x_min 
        x_axis_1.x_max
        x_axis_1.axis_length_on_canvas




        self.wait()







        # Create main surface
        surface = ParametricSurface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )


        # Create the surface
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 1.0],
            height=5,
            width=5,
            depth=3.5,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )
        
        # Add labels
        x_label = Tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        y_label = Tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, RIGHT)
        y_label.next_to(axes.y_axis, UP)
        z_label.next_to(axes.z_axis, OUT)
        z_label.rotate(90*DEGREES, [1,0,0])
        
        ts = TexturedSurface(surface, '_2025/backprop_1/p_24_28_losses_4.png')
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        
        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)
        self.add(u_gridlines)
        self.add(v_gridlines)
        self.add(ts)


        self.frame.reorient(32, 59, 0, (1.88, 1.0, 1.52), 7.78)

        
        self.embed()
        self.wait(20)





