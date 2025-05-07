from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets') #hacks
from welch_axes import *
from functools import partial
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

MAX_RENDERS=2

data_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/poster/gpu_renders_2/'
alphas_1=np.linspace(-2.5, 2.5, 512)

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

def param_surface_3(u, v, surf_array, scaling=0.07, surf_mean=0):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = scaling*(surf_array[v_idx, u_idx]-surf_mean) #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def get_pivot_and_scale(axis_min, axis_max, axis_end):
    '''Above collapses into scaling around a single pivot when axis_start=0'''
    scale = axis_end / (axis_max - axis_min)
    return axis_min, scale


class PosterHackinThree(InteractiveScene):
    def construct(self):
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        paths=[]
        texture_paths=[]
        print('Loading Surface Arrays...')

        render_paths=glob.glob(data_dir+'*')
        for r in render_paths:
        # r=render_paths[0]
            for np_path in glob.glob(r+'/*.npy'):
                # print(np_path)
                loss_arrays.append(np.load(np_path))
                print(np_path, loss_arrays[-1].shape)
                paths.append(np_path)
                plt.clf()
                plt.figure(frameon=False)
                ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                ax.set_axis_off()
                plt.gcf().add_axes(ax)
                plt.imshow(np.rot90(loss_arrays[-1].T)) #have to transpose if transposing u and v and param_surface_1
                texture_paths.append(r+'/'+np_path.split('/')[-1].split('.')[0]+'_texture.png')
                plt.savefig(texture_paths[-1], bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()

        self.wait()

        surfaces=Group()
        surf_functions=[] #Need this later to move dot around.
        grids=Group()
        texts=Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(len(paths[:MAX_RENDERS]))): #LIMIT HERE FOR TESTING
            if 'llama' in paths[i]: scaling=0.07
            elif 'gpt' in paths[i]: scaling=0.2
            elif 'qwen' in paths[i]: scaling=0.07
            elif 'gemma' in paths[i]: scaling=0.07
            else: scaling=0.07

            surf_func=partial(param_surface_3, surf_array=loss_arrays[i], scaling=scaling, surf_mean=np.mean(loss_arrays[i]))
            surf_functions.append(surf_func)
            surface = ParametricSurface(
                surf_func,  
                u_range=[-2.5, 2.5], #Some have different ranges, but I don't think it matters?
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = TexturedSurface(surface, texture_paths[i])
            ts2.set_shading(0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number  of points per line
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

            t=Text(paths[i][-40:])
            t.scale(0.4)
            t.rotate(DEGREES*180)
            t.move_to([0, 3, 0])
            texts.add(t)

        # self.frame.reorient(134, 30, 0, (0.12, 0.11, -0.15), 7.56)
        # self.frame.reorient(132, 30, 0, (0.32, 0.4, -0.32), 6.82)
        self.frame.reorient(132, 30, 0, (0.24, 0.27, -0.23), 7.50)
        self.add(surfaces[0])
        self.add(grids[0])
        self.add(texts[0])
        self.wait()

        for i in range(1, np.min([len(paths), MAX_RENDERS])):
            self.remove(surfaces[i-1])
            self.remove(grids[i-1])
            self.remove(texts[i-1])
            self.add(surfaces[i])
            self.add(grids[i])
            self.add(texts[i])
            self.wait()


        # self.frame.reorient(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        # # self.wait()

        # self.frame.reorient(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        # self.wait()


        self.wait(20)
        self.embed()


class PosterHackinExploreColormaps(InteractiveScene):
    def construct(self):
    
        for cmap in ['viridis', 'plasma']:
            #Load up other surfaces to visualize
            loss_arrays=[]
            paths=[]
            texture_paths=[]
            print('Loading Surface Arrays...')

            render_paths=glob.glob(data_dir+'*')
            for r in render_paths:
            # r=render_paths[0]
                for np_path in glob.glob(r+'/*.npy'):
                    # print(np_path)
                    loss_arrays.append(np.load(np_path))
                    print(np_path, loss_arrays[-1].shape)
                    paths.append(np_path)
                    plt.clf()
                    plt.figure(frameon=False)
                    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                    ax.set_axis_off()
                    plt.gcf().add_axes(ax)
                    plt.imshow(np.rot90(loss_arrays[-1].T), cmap=cmap) #have to transpose if transposing u and v and param_surface_1
                    texture_paths.append(r+'/'+np_path.split('/')[-1].split('.')[0]+'_'+cmap+'_texture.png')
                    plt.savefig(texture_paths[-1], bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

            self.wait()

            surfaces=Group()
            surf_functions=[] #Need this later to move dot around.
            grids=Group()
            texts=Group()
            print("Loading Surfaces and Gridlines...")
            for i in tqdm(range(len(paths[:MAX_RENDERS]))): #LIMIT HERE FOR TESTING
                if 'llama' in paths[i]: scaling=0.07
                elif 'gpt' in paths[i]: scaling=0.2
                elif 'qwen' in paths[i]: scaling=0.07
                elif 'gemma' in paths[i]: scaling=0.07
                else: scaling=0.07

                surf_func=partial(param_surface_3, surf_array=loss_arrays[i], scaling=scaling, surf_mean=np.mean(loss_arrays[i]))
                surf_functions.append(surf_func)
                surface = ParametricSurface(
                    surf_func,  
                    u_range=[-2.5, 2.5], #Some have different ranges, but I don't think it matters?
                    v_range=[-2.5, 2.5],
                    resolution=(512, 512),
                )

                ts2 = TexturedSurface(surface, texture_paths[i])
                ts2.set_shading(0.0, 0.1, 0)
                surfaces.add(ts2)

                num_lines = 64  # Number of gridlines in each direction
                num_points = 512  # Number  of points per line
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

                t=Text(paths[i][-40:])
                t.scale(0.4)
                t.rotate(DEGREES*180)
                t.move_to([0, 3, 0])
                texts.add(t)

            # self.frame.reorient(134, 30, 0, (0.12, 0.11, -0.15), 7.56)
            # self.frame.reorient(132, 30, 0, (0.32, 0.4, -0.32), 6.82)
            self.frame.reorient(132, 30, 0, (0.24, 0.27, -0.23), 7.50)
            self.add(surfaces[0])
            self.add(grids[0])
            self.add(texts[0])
            self.wait()

            for i in range(1, np.min([len(paths), MAX_RENDERS])):
                self.remove(surfaces[i-1])
                self.remove(grids[i-1])
                self.remove(texts[i-1])
                self.add(surfaces[i])
                self.add(grids[i])
                self.add(texts[i])
                self.wait()

            self.remove(surfaces[i])
            self.remove(grids[i])
            self.remove(texts[i])



        # self.frame.reorient(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        # # self.wait()

        # self.frame.reorient(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        # self.wait()


        # self.wait(20)
        self.embed()

