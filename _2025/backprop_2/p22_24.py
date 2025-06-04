from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial
import numpy as np
import torch
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim'
data_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/hackin'


def format_number(num, total_chars=6, align='right'):
    """
    Format number to maintain consistent visual alignment for animations.
    
    Args:
        num: The number to format
        total_chars: Total character width (should accommodate largest expected number)
        align: 'right', 'left', or 'center' - how to align within the fixed width
    """
    abs_num = abs(num)
    
    # Determine appropriate precision based on magnitude
    if abs_num >= 100:
        # 100+: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: two decimal places (e.g., "0.12", "-0.12")
        formatted = f"{num:.2f}"
    
    # Pad to consistent width
    if align == 'right':
        return formatted.rjust(total_chars)
    elif align == 'left':
        return formatted.ljust(total_chars)
    else:  # center
        return formatted.center(total_chars)

def format_number_fixed_decimal(num, decimal_places=2, total_chars=6):
    """
    Alternative formatter that keeps decimal point in same position.
    Useful when you want all numbers to have the same decimal precision.
    """
    formatted = f"{num:.{decimal_places}f}"
    return formatted.rjust(total_chars)

def get_numbers_2(x, w, logits, yhats):

    numbers = VGroup()

    tx = Tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    numbers.add(tx)
    
    # Weights - using consistent formatting
    tm1 = Tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1.scale(0.16)
    tm1.move_to([-1.195, 0.205, 0])
    numbers.add(tm1)
    
    tm2 = Tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
    tm2.scale(0.15)
    tm2.move_to([-1.155, 0.015, 0])
    numbers.add(tm2)
    
    tm3 = Tex(format_number(w[2], total_chars=6)).set_color(GREEN)
    tm3.scale(0.16)
    tm3.move_to([-1.19, -0.17, 0])
    numbers.add(tm3)
    
    # Biases
    tb1 = Tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.16)
    tb1.move_to([-0.875, 0.365, 0])
    numbers.add(tb1)
    
    tb2 = Tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.16)
    tb2.move_to([-0.875, 0.015, 0])
    numbers.add(tb2)
    
    tb3 = Tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    tb3.scale(0.16)
    tb3.move_to([-0.88, -0.335, 0])
    numbers.add(tb3)
    
    # Logits
    tl1 = Tex(format_number(logits[0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    numbers.add(tl1)
    
    tl2 = Tex(format_number(logits[1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    numbers.add(tl2)
    
    tl3 = Tex(format_number(logits[2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.16)  
    tl3.move_to([-0.52, -0.335, 0])
    numbers.add(tl3)
    
    # Predictions
    yhat1 = Tex(f"{yhats[0]:.3f}").set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = Tex(f"{yhats[1]:.3f}").set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = Tex(f"{yhats[2]:.3f}").set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    return numbers



class p22_24(InteractiveScene):
    def construct(self):
        '''
        
        '''
        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.frame.reorient(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        # self.frame.reorient(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)
        # self.frame.reorient(0, 0, 0, (-0.31, -0.02, 0.0), 1.62)
        self.frame.reorient(0, 0, 0, (0.0, 0.07, 0.0), 1.97)

        data=np.load(data_path+'/cities_1d_2.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]

        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.34, 0, -2.34]
        yhats=[0.905, 0.086, 0.008]
        nums=get_numbers_2(x, w, logits, yhats)
        self.add(nums)

        net_background.shift([0,0.23,0])
        nums.shift([0,.23,0])

        # self.wait()

        
        layers=VGroup()
        for p in sorted(glob.glob(svg_path+'/p22_24/*.svg')):
            layers.add(SVGMobject(p)[1:])  
            
        self.add(layers[:7])
        # self.wait()  

        m3_label=net_background[40:42].copy()
        m3_label.scale(1.2)
        m3_label.move_to([1.48, -0.67, 0])
        self.add(m3_label)

        loss_label=layers[1][:4].copy()
        loss_label.scale(0.75)
        loss_label.move_to([[0.7, 0.05, 0]])
        self.add(loss_label)

        yhat2 = Tex("0.086").set_color(YELLOW)
        yhat2.scale(0.14)
        yhat2.move_to([1.205, 0.18, 0])
        self.add(yhat2)

        loss = Tex(format_number(2.45, total_chars=6)).set_color(YELLOW)
        loss.scale(0.16)
        loss.move_to([1.5, 0.18, 0])
        self.add(loss)

        loss_copy = loss.copy()
        loss_copy.move_to([0.7, -0.05, 0])
        self.add(loss_copy)

        m0=nums[2].copy()
        m0.move_to([0.79, -0.68, 0])
        self.add(m0)

        self.wait()


        #DON'T FORGET TO UPDATE ALL PROBS when we change the one wieght!




        # box = SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        # self.play(ShowCreation(box))
        # self.wait()




















        self.wait()
        self.embed()
