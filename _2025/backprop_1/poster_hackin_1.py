from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets') #hacks
from welch_axes import *
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

loss_curve_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_5/all_execpt_embedding_random_64.npy')
loss_curve_2=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_5/all_execpt_embedding_random_51.npy')
loss_curve_3=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_19.npy')
loss_curve_4=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_27.npy')

wormhole_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/wormhole_merged/'
alphas_1=np.linspace(-2.5, 2.5, 512)
loss_2d_1=np.load(wormhole_dir+'000.npy')

class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


##Only need to run this when the underlying npy file changes
# import matplotlib.pyplot as plt
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)) #have to transpose if transposing u and v and param_surface_1
# plt.savefig(wormhole_dir+'loss_2d_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# plt.clf()
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)[128:-128, 128:-128])
# plt.savefig('loss_2d_1_inner.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# Comman for rendering all scenes
# manimgl /Users/stephen/manim/videos/_2025/backprop_1/p39_48.py P39_47 P48_moving_view_1 P48_moving_view_2 P48_moving_view_3 P48_fixed_view -w


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


class PosterHackinTwo(InteractiveScene):
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
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


        self.frame.reorient(90, 0, 0, (0.01, 0.03, -0.0), 6.45) #overhead
        self.wait()

        self.frame.reorient(90, 20, 0, (-0.09, 0.02, 0.04), 6.45) #Slightly down
        self.wait()
        # self.add(s2)
        self.frame.reorient(89, 32, 0, (-0.1, -0.0, 0.05), 5.89) #Moar down
        self.wait()

        self.frame.reorient(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        self.wait()

        self.frame.reorient(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        self.wait()


        #Fixed orentation
        # self.frame.reorient(132, 28, 0, (-0.12, -0.56, 0.33), 4.50) #Kinda wide, but nice I think, could do a closer one too
        # self.frame.reorient(159, 35, 0, (-0.03, -0.62, 0.65), 1.95) #Match with long render
        # self.frame.reorient(180, 23, 0, (-0.06, 0.09, 0.43), 5.81) #Fixed wide veiew I can use at the beginning of 48
        # self.frame.reorient(135, 47, 0, (0.15, 0.28, -0.04), 5.61) #V2 View for side-by-side in p49
        # self.frame.reorient(141, 33, 0, (0.03, 0.21, 0.1), 7.22)

        self.wait()
        self.embed()

        # surface_update_counter=1
        # # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        # self.wait()
        # for i in range(1, len(surfaces)):
        #     # print(i, len(interp_orientations))

        #     self.remove(surfaces[surface_update_counter-1])
        #     self.remove(grids[surface_update_counter-1])
        #     self.add(surfaces[surface_update_counter])
        #     self.add(grids[surface_update_counter])
        #     # print('surface_update_counter', surface_update_counter)

        #     new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        #     s2.move_to(new_point_coords) #This should make point move down smoothly. 
        #     surface_update_counter+=1
        #     self.wait(0.1)

        # self.wait()

        # self.wait()
        # self.play(self.frame.animate.reorient(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.frame.animate.reorient(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.frame.animate.reorient(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.frame.animate.reorient(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.frame.animate.reorient(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        self.embed()