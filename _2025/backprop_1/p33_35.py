from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

surf=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.npy')
xy=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4xy.npy')
grads_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_grads_1_2.npy') 
grads_2=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_grads_2_2.npy') 
xy_grads=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_xy_2.npy') 


def param_surface(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = surf[u_idx, v_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_scaled(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = 1.5*surf[u_idx, v_idx] #Scale for slightly more dramatic 3d viz. 
    except IndexError:
        z = 0
    return np.array([u, v, z])


def get_grads(u,v):
    u_idx = np.abs(xy_grads[0] - u).argmin()
    v_idx = np.abs(xy_grads[1] - v).argmin()
    try:
        z1 = grads_1[u_idx, v_idx]
    except IndexError:
        z1 = 0
    try:
        z2 = grads_2[u_idx, v_idx]
    except IndexError:
        z2 = 0
    return np.array([u, v, z1, z2])


def map_to_canvas(value, axis_min, axis_max, axis_end, axis_start=0):
    value_scaled=(value-axis_min)/(axis_max-axis_min)
    return (value_scaled+axis_start)*axis_end

def get_pivot_and_scale(axis_min, axis_max, axis_end):
    '''Above collapses into scaling around a single pivot when axis_start=0'''
    scale = axis_end / (axis_max - axis_min)
    return axis_min, scale

class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


class P34v2D(InteractiveScene):
    def construct(self):
        '''
        Let's begin by running 2d numerical-ish gradient descent and visiaulizing it in 2 1d panes
        Curves should update as we move in the 2d space. 
        Ideas where initially developed in p33_35_sketch.py - lots of notes there too. 
        '''
        
        ## REMINDED TO TRY CHANGING ARROWS TO THIN CONNECTING LINES AS AS WORK DOWNHILL. 

        num_steps=10
        learning_rate=1.25
        grad_adjustment_factors_1=[0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6] # Not sure why I need these - only applying to viz - 
        grad_adjustment_factors_2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] # maybe I should use for descent too?
        descent_points=[] 
        arrow_end_points_1=[]
        arrow_end_points_2=[]

        staring_values=param_surface(0, 0)
        # starting_grads=get_grads(0,0)

        # step_x_1=learning_rate*abs(starting_grads[2])
        # step_x_2=learning_rate*abs(starting_grads[3])

        descent_points.append(list(staring_values)) #First point

        for i in range(1, num_steps):
            g=get_grads(descent_points[i-1][0], descent_points[i-1][1]) 
            step_x_1=learning_rate*abs(g[2])
            step_x_2=learning_rate*abs(g[3])
            new_x_1=descent_points[i-1][0]+step_x_1
            new_x_2=descent_points[i-1][1]+step_x_2
            arrow_end_points_1.append([new_x_1, descent_points[i-1][2]+step_x_1*g[2]*grad_adjustment_factors_1[i], 0])
            arrow_end_points_2.append([new_x_2, descent_points[i-1][2]+step_x_2*g[3]*grad_adjustment_factors_2[i], 0])
            descent_points.append([new_x_1, new_x_2, param_surface(new_x_1, new_x_2)])
            # print(g)


        self.embed()
        self.wait(20)



























