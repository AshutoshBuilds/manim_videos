from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

save_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/p50/'
xy=np.linspace(-2.5, 2.5, 256)

def get_numerical_gradient(surface_fn, u, v, epsilon=0.01):
    height = surface_fn(u, v)[2]
    height_du = surface_fn(u + epsilon, v)[2]
    du = (height_du - height) / epsilon
    height_dv = surface_fn(u, v + epsilon)[2]
    dv = (height_dv - height) / epsilon
    return (du, dv)

class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

class P50a(InteractiveScene):
    def construct(self):
        '''
        Ok, this has been like a gruelling 10 day animation process - last big scene here. 
        I need a 2d and 3d view of the same thing -> I don't think there's a need to do 2d/3d transitions at this point
        like it might be nice - but I'm already running behind and I think I can make it pretty clear 
        from 2 differen scenes. 

        So I'll do a 2d and 3d version of the same scene. 
        I think let's do the 3d frist, then the 2d can be little slices of that.  

        Yo claude! I'm trying to make a demonstration "loss landscape" that has some special properties, I'm using python. Can you help me create a 2d numpy array that contains 2 parabaoild shapes? One large/gradual one that takes up the whole space (let's do x=-2.5->2.5 and y=-2.5->2.5), and one small parabaloid that's like a "little local valley"? I don't want the small one in the center.
        '''


        surf=np.load(save_dir+'p50_2d.npy') #Optionally add scaling factor here

        # Create the surface
        axes = ThreeDAxes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
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
            u_idx = np.abs(xy - u).argmin()
            v_idx = np.abs(xy - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = ParametricSurface(
            param_surface,
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(256, 256),
        )
        
        ts = TexturedSurface(surface, save_dir+'p50_2d.png')
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-2.5, 2.5, num_points)
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
        
        # Add everything to the scene
        self.frame.reorient(23, 53, 0, (0.11, 0.34, 0.31), 8.28)
        self.add(axes[:2], x_label, y_label)#, z_label) #Don't really need vertical axis here!
        self.wait()

        self.play(ShowCreation(ts),
                ShowCreation(u_gridlines),
                 ShowCreation(v_gridlines))
        self.wait()

        self.play(self.frame.animate.reorient(27, 41, 0, (-0.07, 0.46, 0.58), 7.36), run_time=4.0)
        self.wait()

        # Alrighty, now I need to visualize slices and run gradient descent
        # I could do analytical, but learning towards just doing numerical with velocity - that worked fine before
        
        slice_1_index=-0.5
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_1_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_1 = VMobject()
        slice_1.set_points_smoothly(points)
        slice_1.set_stroke(width=4, color=YELLOW, opacity=0.8)
        self.wait()

        self.play(ShowCreation(slice_1))
        self.wait()

        starting_coords=[-1,-0.5] 
        starting_point=param_surface(*starting_coords)
        s1=Dot3D(center=starting_point, radius=0.1, color='$FF00FF')
        self.add(s1)


        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01 - play with this for smoother/longer animation
        learning_rate=3e-3
        momentum = 0.95 #lil momentum to get to a decent minima lol
        trajectory=[[starting_point[0], starting_point[1], param_surface(starting_point[0], starting_point[1])[2]]]
        velocity = np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            velocity = momentum * velocity - learning_rate * np.array(g)
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]
            trajectory.append([new_x, new_y, param_surface(new_x, new_y)[2]])

        #I don't think we need/want a camera move here. 
        t = VMobject()
        t.set_stroke(width=6, color="#FF00FF", opacity=1.0)
        self.add(t)
        for i in range(num_steps):
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.frame.reorient(*interp_orientations[i])
            self.wait(0.1)
        self.wait()



        self.wait(20)
        self.embed()












