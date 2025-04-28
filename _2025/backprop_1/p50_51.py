from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets') #hacks
from welch_axes import *

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


def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """
    Linearly interpolate between two camera orientations.
    
    Parameters:
    - start_orientation: List containing camera parameters with a tuple at index 3
    - end_orientation: List containing camera parameters with a tuple at index 3
    - num_steps: Number of interpolation steps (including start and end)
    
    Returns:
    - List of interpolated orientations
    """
    result = []
    
    for step in range(num_steps):
        # Calculate interpolation factor (0 to 1)
        t = step / (num_steps - 1) if num_steps > 1 else 0
        
        # Create a new orientation for this step
        interpolated = []
        
        for i in range(len(start_orientation)):
            if i == 3:  # Handle the tuple at position 3
                start_tuple = start_orientation[i]
                end_tuple = end_orientation[i]
                
                # Interpolate each element of the tuple
                interpolated_tuple = tuple(
                    start_tuple[j] + t * (end_tuple[j] - start_tuple[j])
                    for j in range(len(start_tuple))
                )
                
                interpolated.append(interpolated_tuple)
            else:  # Handle regular numeric values
                start_val = start_orientation[i]
                end_val = end_orientation[i]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated.append(interpolated_val)
        
        result.append(interpolated)
    
    return result

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

        self.play(self.frame.animate.reorient(31, 44, 0, (-0.03, 0.45, 0.53), 7.36), run_time=4.0)
        self.wait()

        # Alrighty, now I need to visualize slices and run gradient descent
        # I could do analytical, but learning towards just doing numerical with velocity - that worked fine before
        
        slice_1_index=-0.5
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_1_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_1 = VMobject()
        slice_1.set_points_smoothly(points)
        slice_1.set_stroke(width=4, color=YELLOW, opacity=1.0)
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
        #I think add slices every N steps - maybe with little bit different opacities?

        way_point_1=48
        way_point_2=71
        way_point_3=96

        t = VMobject()
        t.set_stroke(width=5, color="#FF00FF", opacity=0.9)
        self.add(t)
        for i in range(way_point_1): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            # self.wait(0.1)
        self.wait()

        slice_2_index=trajectory[way_point_1][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_2_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_2 = VMobject()
        slice_2.set_points_smoothly(points)
        slice_2.set_stroke(width=4, color=YELLOW, opacity=0.7)
        self.wait()

        self.play(ShowCreation(slice_2))
        self.wait()

        for i in range(way_point_1, way_point_2): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            # self.wait(0.1)
        self.wait()

        slice_3_index=trajectory[way_point_2][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_3_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_3 = VMobject()
        slice_3.set_points_smoothly(points)
        slice_3.set_stroke(width=4, color=YELLOW, opacity=0.5)
        self.wait()

        self.play(ShowCreation(slice_3))
        self.wait()


        start_orientation=[31, 44, 0, (-0.03, 0.45, 0.53), 7.36]
        end_orientation=[38, 37, 0, (-0.0, 0.47, 0.53), 7.94]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=way_point_3-way_point_2)
        self.wait()

        for i in range(way_point_2, way_point_3): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.frame.reorient(*interp_orientations[i-way_point_2])
            # self.wait(0.1)
        self.wait()


        slice_4_index=trajectory[way_point_3][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_4_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_4 = VMobject()
        slice_4.set_points_smoothly(points)
        slice_4.set_stroke(width=4, color=YELLOW, opacity=0.3)
        self.wait()

        self.play(ShowCreation(slice_4))
        self.wait()


        # self.remove(ts)


        ## Hmm why is this fucker changing color?
        # dot_path=Group()
        # self.add(dot_path)
        # for i in range(num_steps): #First leg -> getting stuck
        #     s1.move_to(trajectory[i]) #Get above surface to not fuck up opacity
        #     dot_path.add(Dot3D(center=trajectory[i], radius=0.03, color='$FF00FF').set_opacity(0.8))
        #     # self.wait(0.1)
        # self.wait()

        # dot_path.set_opacity(0.5)
        # self.add(dot_path)
        # self.remove(dot_path)

        # Is it crazy to clear everything and to the 2d animation in the same sequence? 
        # Let's see here...

        self.remove(slice_1, slice_2, slice_3, slice_4, ts, u_gridlines, v_gridlines, axes, s1, t, x_label, y_label)
        # self.frame.reorient(0, 0, 0, (0,0,0), 8.00)
        self.frame.reorient(0, 0, 0, (5.16, 1.19, -0.0), 7.14)
        self.wait()

        x_axis_1=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=-1, y_max=2, y_ticks=[], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_1 = Tex(r'\theta_1', font_size=28).set_color(CHILL_BROWN)
        y_label_1 = Tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, UP, buff=0.08)


        axes_1=VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.add(axes_1)
        self.wait()

        points_1 = [param_surface(u, slice_1_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_1))
        curve_1=VMobject()         
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)
        self.play(ShowCreation(curve_1))
        self.wait()


        points_2 = [param_surface(u, slice_2_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=VMobject()         
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.7)
        self.play(ShowCreation(curve_2))
        self.wait()

        points_2 = [param_surface(u, slice_3_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=VMobject()         
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.5)
        self.play(ShowCreation(curve_2))
        self.wait()

        points_2 = [param_surface(u, slice_4_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=VMobject()         
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.3)
        self.play(ShowCreation(curve_2))
        self.wait()



        self.wait(20)
        self.embed()












