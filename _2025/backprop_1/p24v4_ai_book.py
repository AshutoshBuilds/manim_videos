from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
WELCH_RED='#EC2027'
BLUE='#65c8d0'


class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


class P24v4(InteractiveScene):
    def construct(self):

        surf=2.0*np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4.npy')+0.8
        xy=np.load('/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4xy.npy')

        # Create the surface
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 0.6],
            height=5,
            width=5,
            depth=4,
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
            resolution=(512, 512),
        )
        
        ts = TexturedSurface(surface, '/home/zedaes/Documents/Welch Labs/ai_book/backprop/data/p_24_28_losses_4.png')
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 1024  # Number of points per line
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
        self.frame.reorient(-24, 60, 0, (1.71, 1.77, 2.21), 8.03)
        self.wait()
        self.play(ShowCreation(u_gridlines), ShowCreation(v_gridlines), self.frame.animate.reorient(38, 61, 0, (1.33, 1.43, 2.61), 7.78), run_time=4)
        self.play(FadeIn(ts))
        self.wait()

        #Ok now we want to move to and draw the curve for the first parameter projection. 
        self.play(self.frame.animate.reorient(0, 91, 0, (1.71, 1.05, 3.47), 6.46), 
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
        slice_1.set_stroke(width=4, color=WELCH_RED, opacity=0.8)

        self.play(ShowCreation(slice_1), run_time=2)
        self.wait()

        slice_t_bottom=Dot(param_surface(1.61, 0), radius=0.08, fill_color=WELCH_RED)
        slice_t_bottom.rotate(90*DEGREES, [1,0,0])
        self.add(slice_t_bottom)
        self.wait()

        y_label.rotate(90*DEGREES, [0,0,1]) #Maybe some way to hide this jump? Can't figure out how to do it in one step
        self.play(self.frame.animate.reorient(79, 98, 0, (1.71, 1.59, 3.29), 6.21),
                  y_label.animate.rotate(90*DEGREES, [0,1,0]),
                  run_time=4)
        self.wait()
        

        v_points = np.linspace(-1, 4, num_points)
        points = [param_surface(1.61, v) for v in v_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_2 = VMobject()
        slice_2.set_points_smoothly(points)
        slice_2.set_stroke(width=4, color=BLUE, opacity=0.8)   

        self.play(ShowCreation(slice_2), run_time=2)
        self.wait()


        #note sure about camera movement here yet. Maybe camear moves while I draw the second curve? while making curve more opaque?
        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 1.76) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_3 = VMobject()
        slice_3.set_points_smoothly(points)
        slice_3.set_stroke(width=4, color=WELCH_RED, opacity=0.8)
        self.wait()

        slice_bottom_2=Dot(param_surface(1.61, 1.76), radius=0.08, fill_color=BLUE)
        slice_bottom_2.rotate(90*DEGREES, [1,0,0])
        slice_bottom_2.rotate(90*DEGREES, [0,0,1])
        self.add(slice_bottom_2)
        self.wait()

        #Ok i think it makes sense to quickly draw then then move the camera. 
        self.play(ShowCreation(slice_3))

        self.wait()
        self.play(self.frame.animate.reorient(1, 96, 0, (1.4, 1.57, 3.52), 6.57), 
                  slice_1.animate.set_stroke(opacity=0.4), run_time=4)
        self.wait()

        slice_bottom_3=Dot(param_surface(1.11, 1.76), radius=0.08, fill_color=WELCH_RED)
        slice_bottom_3.rotate(90*DEGREES, [1,0,0])
        self.add(slice_bottom_3)
        self.wait()

        self.play(self.frame.animate.reorient(42, 46, 0, (1.5, 1.67, 2.4), 7.61),
                 ts.animate.set_opacity(0.6),
                 u_gridlines.animate.set_stroke(opacity=0.2),
                 v_gridlines.animate.set_stroke(opacity=0.2), run_time=4)
        self.wait()

        self.play(
            slice_1.animate.set_opacity(0.0),
            slice_2.animate.set_opacity(0.0),
            slice_3.animate.set_opacity(0.0),
            slice_bottom_2.animate.set_opacity(0.0),
            slice_t_bottom.animate.set_opacity(0.0),
            slice_bottom_3.animate.set_opacity(0.0),
            self.frame.animate.reorient(0, 49, 0, (1.48, 1.79, 2.54), 7.61),
            run_time=6
        )
        self.wait()


        #Bottom of Bowl
        u_min=0.95 
        v_min=2.3
        global_min=Dot3D(center=param_surface(u_min,v_min), radius=0.1, color='$FF00FF')
        self.add(global_min)
        self.wait()

        self.play(ts.animate.set_opacity(0.0), 
                  u_gridlines.animate.set_opacity(0.0),
                  v_gridlines.animate.set_opacity(0.0),
                  x_label.animate.rotate(-90*DEGREES, [1,0,0]),
                  y_label.animate.rotate(-90*DEGREES, [0,1,0]).rotate(-90*DEGREES, [0,0,1]),
                  self.frame.animate.reorient(0, 0, 0, (1.54, 1.54, 2.52), 5.59), #overhead
                  run_time=6)





        # self.play(self.frame.animate.reorient(-70, 49, 0, (1.59, 1.52, 2.39), 6.84), run_time=10, rate_func=linear)
        self.wait()




        self.embed()
        self.wait(20)