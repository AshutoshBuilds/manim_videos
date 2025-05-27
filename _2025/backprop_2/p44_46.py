from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial
import numpy as np
import torch

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim'
data_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/hackin'


def format_number(num, total_chars=5):
    """
    Format number to always show same total character width including sign.
    
    Args:
        num: The number to format
        total_chars: Total character width including decimal point and sign
    """
    abs_num = abs(num)
    sign = '-' if num < 0 else ''
    
    # Account for sign in character count
    available_chars = total_chars - len(sign)
    
    if abs_num >= 1000:
        # For very large numbers, use scientific notation or integer
        formatted = f"{num:.0f}"
    elif abs_num >= 100:
        # 100-999: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: adjust decimal places based on available space
        if available_chars >= 4:  # Room for "0.xx"
            formatted = f"{num:.2f}"
        else:
            formatted = f"{num:.1f}"
    
    return formatted

class p44_v1(InteractiveScene):
    def construct(self):

        data=np.load(data_path+'/cities_1d_1.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.frame.reorient(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        self.frame.reorient(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)

        #Ok, let's render some numbers
        i=3
        x=xs[i, -1]
        tx = Tex(str(x) + r'^\circ')
        tx.scale(0.13)
        tx.move_to([-1.49, 0.02, 0])
        self.add(tx)

        w=weights[i,:]
        tm1=Tex(format_number(w[0])).set_color('#00FFFF')
        tm1.scale(0.16)
        tm1.move_to([-1.195, 0.205, 0])
        self.add(tm1)

        tm2=Tex(format_number(w[1])).set_color(YELLOW)
        tm2.scale(0.15)
        tm2.move_to([-1.155, 0.015, 0])
        self.add(tm2)

        tm3=Tex(format_number(w[2])).set_color(GREEN)
        tm3.scale(0.16)
        tm3.move_to([-1.19, -0.17, 0])
        self.add(tm3)

        tb1=Tex(format_number(w[3])).set_color('#00FFFF')
        tb1.scale(0.16)
        tb1.move_to([-0.875, 0.365, 0])
        self.add(tb1)

        tb2=Tex(format_number(w[4])).set_color(YELLOW)
        tb2.scale(0.16)
        tb2.move_to([-0.875, 0.015, 0])
        self.add(tb2)

        tb3=Tex(format_number(w[5])).set_color(GREEN)
        tb3.scale(0.16)
        tb3.move_to([-0.88, -0.335, 0])
        self.add(tb3)

        tl1=Tex(format_number(logits[i,0])).set_color('#00FFFF')
        tl1.scale(0.16)
        tl1.move_to([-0.52, 0.37, 0])
        self.add(tl1)

        tl2=Tex(format_number(logits[i,1])).set_color(YELLOW)
        tl2.scale(0.16)
        tl2.move_to([-0.52, 0.015, 0])
        self.add(tl2)

        tl3=Tex(format_number(logits[i,2])).set_color(GREEN)
        tl3.scale(0.16)
        tl3.move_to([-0.52, -0.335, 0])
        self.add(tl3)

        yhat1=Tex(format_number(yhats[i,0])).set_color('#00FFFF')
        yhat1.scale(0.16)
        yhat1.move_to([0.22, 0.37, 0])
        self.add(yhat1)

        yhat2=Tex(format_number(yhats[i,1])).set_color(YELLOW)
        yhat2.scale(0.16)
        yhat2.move_to([0.22, 0.015, 0])
        self.add(yhat2)

        yhat3=Tex(format_number(yhats[i,2])).set_color(GREEN)
        yhat3.scale(0.16)
        yhat3.move_to([0.22, -0.335, 0])
        self.add(yhat3)

        #Ok let's shade some lines!
        max_region_width=0.15
        min_region_width=0.01
        region_scaling=0.15

        y_one_hot=torch.nn.functional.one_hot(torch.tensor(int(ys[i])),3).numpy()
        dldh=yhats[i]-y_one_hot

        rh1_width=np.clip(region_scaling*np.abs(dldh[0]), min_region_width, max_region_width)
        rh1=Rectangle(0.425, rh1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
        rh1.move_to([-0.52, 0.37, 0])
        self.add(rh1)

        rh2_width=np.clip(region_scaling*np.abs(dldh[1]), min_region_width, max_region_width)
        rh2=Rectangle(0.425, rh2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
        rh2.move_to([-0.52, 0.015, 0])
        self.add(rh2)

        rh3_width=np.clip(region_scaling*np.abs(dldh[2]), min_region_width, max_region_width)
        rh3=Rectangle(0.425, rh3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
        rh3.move_to([-0.52, -0.335, 0])
        self.add(rh3)


        self.wait()
        self.embed()

















