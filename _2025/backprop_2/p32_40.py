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
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'

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

def get_numbers_3(x, w, logits, yhats):

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
    yhat1 = Tex(format_number(yhats[0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = Tex(format_number(yhats[1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = Tex(format_number(yhats[2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    
    return numbers


class p32_40(InteractiveScene):
    def construct(self):
        '''
        
        '''
        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.34, 0, -2.34]
        yhats=[0.91, 0.09, 0.00]
        nums=get_numbers_3(x, w, logits, yhats)
        
        rect_1=RoundedRectangle(0.65, 1.1, 0.02)
        rect_1.set_stroke(width=0)
        rect_1.set_color(CHILL_GREEN)
        rect_1.set_opacity(0.4)
        rect_1.move_to([-1.0, 0, 0])

        rect_2=RoundedRectangle(0.54, 1.1, 0.02)
        rect_2.set_stroke(width=0)
        rect_2.set_color(CHILL_BLUE)
        rect_2.set_opacity(0.4)
        rect_2.move_to([-0.15, 0.0, 0])

        dldm=Tex(r"\frac{\partial L}{\partial m}")
        modular_eq_1=Tex("=")
        modular_eq_2=Tex(r"\frac{\partial h}{\partial m}")
        modular_eq_3=Tex(r"\cdot")
        modular_eq_4=Tex(r"\frac{\partial L}{\partial h}")

        modular_eq=VGroup(dldm, modular_eq_1, modular_eq_2, modular_eq_3, modular_eq_4)
        dldm.scale(0.17)
        modular_eq_1.scale(0.22)
        modular_eq_2.scale(0.17)
        modular_eq_3.scale(0.35)
        modular_eq_4.scale(0.17)
        dldm.move_to([-1.5, -0.7, 0])
        modular_eq_1.next_to(dldm, RIGHT, buff=0.18)
        modular_eq_2.next_to(modular_eq_1, RIGHT, buff=0.15)
        modular_eq_3.next_to(modular_eq_2, RIGHT, buff=0.4)
        modular_eq_4.next_to(modular_eq_3, RIGHT, buff=0.35)


        #p32c animation
        self.frame.reorient(0, 0, 0, (-0.62, -0.1, 0.0), 1.56)
        self.add(net_background, nums)
        self.wait(0)
        self.play(FadeIn(dldm))
        self.play(FadeIn(rect_1), ReplacementTransform(dldm[3:].copy(), modular_eq_2[2:]))
        self.add(modular_eq_2, modular_eq_1)
        self.wait()
        self.play(FadeIn(rect_2), ReplacementTransform(dldm[:2].copy(), modular_eq_4[:2]))
        self.add(modular_eq_3, modular_eq_4)
        self.wait()


        #p33
        softmax_box=VGroup(net_background[18:34], net_background[11])
        not_softmax_box=VGroup(net_background[:11], net_background[12:18], net_background[34:])

        # self.remove(not_softmax_box)
        softmax_derivative=SVGMobject(svg_path+'/p33_to_manim.svg')[1:]
        softmax_derivative.scale(1.17)
        softmax_derivative.move_to([1.4,-0.07,0])

        dldh_eq=Tex(r"\frac{\partial L}{\partial h}=\hat{y}-y")
        dldh_eq.scale(0.25)
        dldh_eq.move_to([1.28, -0.95, 0])
        dldh_border=RoundedRectangle(dldh_eq.get_width()+0.1, dldh_eq.get_height()+0.1, 0.02).move_to(dldh_eq.get_center())
        dldh_border.set_color(YELLOW).set_stroke(width=2)

        #p33a animation
        self.wait()
        self.remove(rect_1, rect_2, not_softmax_box)
        self.play(nums.animate.set_opacity(0.0), modular_eq[:-1].animate.set_opacity(0.0))
        self.play(ShowCreation(softmax_derivative),
                  # self.frame.animate.reorient(0, 0, 0, (0.86, -0.1, 0.0), 2.95),
                  self.frame.animate.reorient(0, 0, 0, (1.35, -0.11, 0.0), 2.40),
                  ReplacementTransform(modular_eq[-1], dldh_eq[:5]),
                  run_time=3
                  )
        self.add(dldh_eq)
        self.wait()
        self.play(ShowCreation(dldh_border))
        self.wait()


        # self.play(not_softmax_box.animate.set_opacity(0.0), modular_eq[:-1].animate.set_opacity(0.0), nums.animate.set_opacity(0.0))

                 # rect_1.animate.set_opacity(0.0), rect_2.animate.set_opacity(0.0), run_time=2)


        self.wait()


        self.add(dldh_eq, dldh_border)

        self.play(ShowCreation(softmax_derivative), run_time=2.0)


        self.frame.reorient(0, 0, 0, (0.86, -0.1, 0.0), 2.95)
        self.add(softmax_derivative)


        self.wait()
        self.embed()







