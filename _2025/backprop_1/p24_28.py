from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class P24v1(InteractiveScene):
    def construct(self):

        #TODO - render higher rez version
        surf=1.6*np.load('_2025/backprop_1/p_24_28_losses_3.npy') #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load('_2025/backprop_1/p_24_28_losses_3xy.npy')

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

        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = ParametricSurface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )
        
        ts = TexturedSurface(surface, '_2025/backprop_1/p_24_28_losses_3.png')
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

        # self.add(u_gridlines, v_gridlines)
        # self.frame.reorient(10, 57, 0, (2.09, 1.11, 1.36), 7.55)
        self.frame.reorient(-31, 55, 0, (1.94, 0.85, 1.25), 8.40)
        self.wait()
        self.play(ShowCreation(u_gridlines), ShowCreation(v_gridlines), self.frame.animate.reorient(32, 59, 0, (1.88, 1.0, 1.52), 7.78), run_time=4)
        self.play(FadeIn(ts))
        self.wait()

        #Ok now we want to move to and draw the curve for the first parameter projection. 
        self.play(self.frame.animate.reorient(0, 90, 0, (1.64, 1.05, 2.09), 6.46), 
                  x_label.animate.rotate(90*DEGREES, [1,0,0]), 
                  ts.animate.set_opacity(0.2),
                  u_gridlines.animate.set_stroke(opacity=0.15),
                  v_gridlines.animate.set_stroke(opacity=0.15),
                  run_time=3)
        self.wait()

        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 0) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_1 = VMobject()
        slice_1.set_points_smoothly(points)
        slice_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        self.play(ShowCreation(slice_1), run_time=2)
        self.wait()

        slice_t_bottom=Dot(param_surface(1.61, 0), radius=0.08, fill_color=YELLOW)
        slice_t_bottom.rotate(90*DEGREES, [1,0,0])
        self.add(slice_t_bottom)
        self.wait()

        self.play(self.frame.animate.reorient(81, 97, 0, (1.72, 1.73, 1.81), 6.51))
        self.wait()

        v_points = np.linspace(-1, 4, num_points)
        points = [param_surface(1.61, v) for v in v_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_2 = VMobject()
        slice_2.set_points_smoothly(points)
        slice_2.set_stroke(width=4, color=BLUE, opacity=0.8)   

        self.play(ShowCreation(slice_2), run_time=2)
        self.wait()


        self.embed()
        self.wait(20)
        









