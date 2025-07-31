from manimlib import *
from functools import partial
import sys

sys.path.append('_2025/backprop_3') #Point to folder where plane_folding_utils.py is
from geometric_dl_utils import *
from plane_folding_utils import *
from geometric_dl_utils_simplified import *
from polytope_intersection_utils import intersect_polytopes
# from decision_boundary_utils import *

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/' #Point to folder where map images are
colors = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, TEAL]


class p7a(InteractiveScene):
    def construct(self):
        model_path='_2025/backprop_3/models/8_1.pth'
        model = BaarleNet([8])
        model.load_state_dict(torch.load(model_path))
        viz_scales=[0.1, 0.1, 0.05]
        num_neurons=[8, 8, 2]


        