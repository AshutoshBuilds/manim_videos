from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets') #hacks
from welch_axes import *
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

wormhole_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/wormhole_merged/'
alphas_1=np.linspace(-2.5, 2.5, 512)
loss_2d_1=np.load(wormhole_dir+'000.npy')

class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)
def param_surface_1(u, v):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*loss_2d_1[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_2(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def get_pivot_and_scale(axis_min, axis_max, axis_end):
    '''Above collapses into scaling around a single pivot when axis_start=0'''
    scale = axis_end / (axis_max - axis_min)
    return axis_min, scale

def get_numerical_gradient(surface_fn, u, v, epsilon=0.01):
    height = surface_fn(u, v)[2]
    height_du = surface_fn(u + epsilon, v)[2]
    du = (height_du - height) / epsilon
    height_dv = surface_fn(u, v + epsilon)[2]
    dv = (height_dv - height) / epsilon
    return (du, dv)


# Copying code to recreate state
def copy_frame_positioning_precise(frame):
    center = frame.get_center()
    height = frame.get_height()
    angles = frame.get_euler_angles()

    call = f"reorient("
    theta, phi, gamma = (angles / DEG)
    call += f"{theta}, {phi}, {gamma}"
    if any(center != 0):
        call += f", {tuple(center)}"
    if height != FRAME_HEIGHT:
        call += ", {:.2f}".format(height)
    call += ")"
    print(call)
    pyperclip.copy(call)


# copy_frame_positioning_precise(self.frame)

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


class OpeningRewriteFour(InteractiveScene):
    def construct(self):    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=1
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=Group()
        surf_functions=[] #Need this later to move dot around.
        grids=Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = ParametricSurface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = TexturedSurface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            ts2.set_shading(0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = VGroup()
            v_gridlines = VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])

        #Starting semi-overhead
        self.frame.reorient(0, 11, 0, (0.01, 0.03, 0.12), 6.37)
        self.wait()


        #Cinematic pan around
        self.play(self.frame.animate.reorient(91, 44, 0, (-0.15, -0.0, 0.07), 5.78), run_time=10.0)
        self.wait()



        #Zoom in to see point drop in



        # self.play(self.frame.animate.reorient(122, 58, 0, (-1.15, -0.02, 0.11), 5.42), run_time=4.0)
        self.wait()

        starting_coords=[-0.7,0.95]  #[-0.7,1.0]  #
        starting_point=param_surface_1(*starting_coords)
        s1=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
        s1.shift([0,0,1]) #move up so we can "drop it in to skate bro"
        
        self.add(s1)

        self.play(s1.animate.shift([0,0,-1]),
                  # self.frame.animate.reorient(120, 39, 0, (-0.89, 0.27, -0.07), 4.05),
                  run_time=2.0)

        self.wait()

        self.play(self.frame.animate.reorient(124, 37, 0, (-0.96, 0.01, 0.23), 3.41), run_time=4.0)
        self.wait()

        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01 - play with this for smoother/longer animation
        learning_rate=3e-3
        momentum = 0.95 #lil momentum to get to a decent minima lol
        trajectory=[[starting_point[0], starting_point[1], param_surface_1(starting_point[0], starting_point[1])[2]]]
        velocity = np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface_1, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            # delta=learning_rate*np.array(g)
            # new_x=trajectory[-1][0]-delta[0]
            # new_y=trajectory[-1][1]-delta[1]
            # Update velocity with momentum
            velocity = momentum * velocity - learning_rate * np.array(g)
            
            # Update position using velocity
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]

            trajectory.append([new_x, new_y, param_surface_1(new_x, new_y)[2]])

        # num_total_steps=256
        start_orientation=[124, 37, 0, (-0.96, 0.01, 0.23), 3.41]
        end_orientation=[141, 38, 0, (-0.72, 0.13, 0.06), 2.85]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_steps)

        self.wait()

        t = VMobject()
        t.set_stroke(width=6, color="#FF00FF", opacity=1.0)
        self.add(t)
        for i in range(num_steps):
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.frame.reorient(*interp_orientations[i])
            self.wait(0.1)
        self.wait()


        #Move to side for hinton quote!
        # self.play(self.frame.animate.reorient(89, 5, 0, (-0.02, 2.25, 0.0), 6.80), run_time=6.0) #This angling is finicky
        self.play(self.frame.animate.reorient(90.18440887882339, 3.21213142627667, 0.0, (-0.0132118175, 2.2433786, -0.00072974624), 6.80), run_time=6.0)
        # copy_frame_positioning_precise(self.frame)

        self.wait()

        self.play(t.animate.set_opacity(0.0),
                  s1.animate.set_opacity(0.0))
        self.wait()
        self.remove(s1)

        self.play(self.frame.animate.reorient(46, 46, 0, (0.17, -0.16, -0.02), 6.13), run_time=8.0)
        self.wait()


        self.wait(20)
        self.embed()



class ending_moves(InteractiveScene):
    '''
    Hack on single frames. 
    '''
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=66
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        surfaces=Group()
        surf_functions=[] #Need this later to move dot around.
        grids=Group()
        print("Loading Surfaces and Gridlines...")
        
        i=65

        surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
        surf_functions.append(surf_func)
        surface = ParametricSurface(
            surf_func,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        ts2 = TexturedSurface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
        ts2.set_shading(0.0, 0.1, 0)
        surfaces.add(ts2)

        num_lines = 64  # Number of gridlines in each direction
        num_points = 512  # Number of points per line
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        for u in u_values:
            points = [surf_func(u, v) for v in v_points]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [surf_func(u, v) for u in u_points]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.15)
            v_gridlines.add(line)
        grids.add(VGroup(u_gridlines, v_gridlines))


        new_point_coords=surf_func(*starting_coords)
        s2.move_to(new_point_coords)

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        # self.frame.reorient(179, 32, 0, (0.1, -0.42, 1.36), 0.48) #Super tight on wormhole
        self.frame.reorient(-179, 7, 0, (0.1, -0.43, 1.36), 0.48)
        self.wait()
        self.play(self.frame.animate.reorient(-180, 24, 0, (0.05, 0.3, 0.97), 5.17), run_time=12)


        # self.frame.reorient(89, 180, 0, (0.04, -0.22, 1.32), 6.49) #Cool underneath view
        # self.frame.reorient(89, 112, 0, (0.26, -0.03, 0.99), 6.22)
        # self.wait()

        # self.play(self.frame.animate.reorient(180, 82, 0, (0.02, -0.02, 0.97), 6.22), run_time=10)
        # self.wait()

        # self.play(self.frame.animate.reorient(180, 12, 0, (0.12, 0.2, 1.3), 6.22), run_time=15)

        # self.play(self.frame.animate.reorient(-180, 12, 0, (0.12, 0.2, 1.3), 6.22), run_time=25, rate_func=linear)
        self.wait(5)
        self.embed()



