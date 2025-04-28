from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

wormhole_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_27_5/'
alphas_1=np.linspace(-2.5, 2.5, 512)

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

class P48cV1(InteractiveScene):
    def construct(self):
        '''
        Wikitext training example - man I hope this works. 
        '''
    
        #Load up other surfaces to visualize
        loss_arrays_pre=[]
        loss_arrays_post=[]
        loss_arrays_interleaved=[]
        num_time_steps=2
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays_pre.append(np.load(wormhole_dir+'pre_step_'+str(i).zfill(3)+'.npy'))
            loss_arrays_post.append(np.load(wormhole_dir+'post_step_'+str(i).zfill(3)+'.npy'))
            loss_arrays_interleaved.append(loss_arrays_pre[-1])
            loss_arrays_interleaved.append(loss_arrays_post[-1])


        # self.wait()
        # import matplotlib.pyplot as plt
        # data_max=np.array(loss_arrays_interleaved).max() 
        # for i in range(len(loss_arrays_interleaved)):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays_interleaved[i].T), vmax=1.1*data_max) #Artificial color ceiling to make it not all yellow
        #     plt.savefig(wormhole_dir+'loss_arrays_interleaved_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=Group()
        surf_functions=[] #Need this later to move dot around.
        grids=Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(len(loss_arrays_interleaved))):
            surf_func=partial(param_surface_2, surf_array=loss_arrays_interleaved[i])
            surf_functions.append(surf_func)
            surface = ParametricSurface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = TexturedSurface(surface, wormhole_dir+'loss_arrays_interleaved_'+str(i).zfill(3)+'.png')
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


        starting_coords=[0.05,-0.9]
        starting_point=surf_functions[0](*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)

        self.frame.reorient(142, 34, 0, (-0.09, -0.77, 0.15), 3.55)

        
        surface_update_counter=2
        self.remove(surfaces[surface_update_counter-1])
        self.remove(grids[surface_update_counter-1])
        self.add(surfaces[surface_update_counter])
        self.add(grids[surface_update_counter])
        new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        s2.move_to(new_point_coords) 




        num_total_steps=38 #Crank this for final viz
        start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        # end_orientation=[131, 31, 0, (-0.12, -0.88, 0.22), 2.90]
        end_orientation=[122, 11, 0, (-0.01, -0.28, 0.55), 2.05] #Move overhead
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        surface_update_counter=1
        frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        self.wait()
        for i in range(1, num_total_steps):
            # print(i, len(interp_orientations))
            if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):

                self.remove(surfaces[surface_update_counter-1])
                self.remove(grids[surface_update_counter-1])
                self.add(surfaces[surface_update_counter])
                self.add(grids[surface_update_counter])

                new_point_coords=surf_functions[surface_update_counter](*starting_coords)
                s2.move_to(new_point_coords) #This should make point move down smoothly. 
                surface_update_counter+=1

            # print(i, len(interp_orientations))
            self.frame.reorient(*interp_orientations[i])
            self.wait(0.1)

        self.wait()

        self.embed()