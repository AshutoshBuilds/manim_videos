from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

# surf_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.npy')
# xy_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4xy.npy')

loss_curve_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_2/all_execpt_embedding_random_24.npy')
loss_curve_2=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_2/all_execpt_embedding_random_23.npy')

# def param_surface(u, v):
#     u_idx = np.abs(xy[0] - u).argmin()
#     v_idx = np.abs(xy[1] - v).argmin()
#     try:
#         z = surf[u_idx, v_idx]
#     except IndexError:
#         z = 0
#     return np.array([u, v, z])


class P39_48(InteractiveScene):
    def construct(self):
        '''
		Not going to lie I'm kinda terrified of this scene - buuuuut if I can pull of 
		some level of what's in my head I do think it will be DOPE. 
		Ok one step at a time here. Starting with one and then 6 2d panels, then being the last two 
		together into 3d very much like I did in the last big scene. 
        '''


        self.embed()
        self.wait()

