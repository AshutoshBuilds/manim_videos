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
        net_background=SVGMobject(svg_path+'/p44_background_1.svg')[1:]

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


        #p33 animation
        # softmax_box=VGroup(net_background[18:34], net_background[11])
        # not_softmax_box=VGroup(net_background[:11], net_background[12:18], net_background[34:])
        softmax_box=VGroup(net_background[17:33], net_background[10])
        not_softmax_box=VGroup(net_background[:10], net_background[11:17], net_background[33:])

        # self.remove(not_softmax_box)
        softmax_derivative=SVGMobject(svg_path+'/p33_to_manim.svg')[1:]
        softmax_derivative.scale(1.17)
        softmax_derivative.move_to([1.4,-0.07,0])

        dldh_eq=Tex(r"\frac{\partial L}{\partial h}=\hat{y}-y")
        dldh_eq.scale(0.25)
        dldh_eq.move_to([1.28, -0.95, 0])
        dldh_border=RoundedRectangle(dldh_eq.get_width()+0.1, dldh_eq.get_height()+0.1, 0.02).move_to(dldh_eq.get_center())
        dldh_border.set_color(YELLOW).set_stroke(width=2)
        dldh_with_border=VGroup(dldh_eq, dldh_border)

        #p33a animation
        self.wait()
        # self.remove(rect_1, rect_2, not_softmax_box)
        self.play(nums.animate.set_opacity(0.0), modular_eq[:-1].animate.set_opacity(0.0),
                 rect_1.animate.set_opacity(0.0), rect_2.animate.set_opacity(0.0), not_softmax_box.animate.set_opacity(0.0))
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

        #p34 setup
        p34_blocks=VGroup()
        for p in sorted(glob.glob(svg_path+'/p34/*.svg')):
            p34_blocks.add(SVGMobject(p)[1:])  
        p34_blocks[0].move_to([-0.02, 0.01, 0])
        p34_blocks[1].move_to([0.62, 0.54, 0])
        p34_blocks[2].move_to([-0.057, 0.047, 0])
        p34_blocks[4].move_to([1.4, 0.42, 0 ])
        p34_blocks[5].move_to([1.4, 0.0, 0 ])

        # Go ahead and load up modular net background we need for p35
        p35_net_background=VGroup()
        for p in sorted(glob.glob(svg_path+'/p_35/*.svg')):
            p35_net_background.add(SVGMobject(p)[1:])  
        self.wait()

        self.wait()
        self.play(FadeOut(softmax_derivative), run_time=1.0)
        self.play(FadeIn(p35_net_background[:3]),
                  not_softmax_box.animate.set_opacity(1.0), 
                  nums.animate.set_opacity(1.0), 
                  dldh_with_border.animate.scale(0.8).move_to([1.28, 0.7, 0]),
                  self.frame.animate.reorient(0, 0, 0, (-0.02, 0.0, 0.0), 1.90), run_time=2.0)
        self.remove(softmax_box)
        self.add(p34_blocks[0], p34_blocks[1][:-1])
        self.wait(0)

        y1=Tex("0").set_color("#00FFFF").scale(0.19).next_to(nums[-3], RIGHT, buff=0.24)
        y2=Tex("1").set_color(YELLOW).scale(0.19).next_to(nums[-2], RIGHT, buff=0.24)
        y3=Tex("0").set_color(GREEN).scale(0.19).next_to(nums[-1], RIGHT, buff=0.24)

        self.play(FadeIn(y1))
        self.play(FadeIn(y2))
        self.play(FadeIn(y3))
        self.wait()

        self.play(ShowCreation(p34_blocks[2]))
        self.remove(p34_blocks[2])
        self.wait()


        minus_1=Tex("-").scale(0.2).next_to(nums[-3], RIGHT, buff=0.09)
        equals_1=Tex("=").scale(0.2).next_to(y1, RIGHT, buff=0.12)
        d_1=Tex("0.91").set_color("#00FFFF").scale(0.18).next_to(y1, RIGHT, buff=0.28)

        minus_2=Tex("-").scale(0.2).next_to(nums[-2], RIGHT, buff=0.09)
        equals_2=Tex("=").scale(0.2).next_to(y2, RIGHT, buff=0.12)
        d_2=Tex("-0.91").set_color(YELLOW).scale(0.18).next_to(y2, RIGHT, buff=0.28)

        minus_3=Tex("-").scale(0.2).next_to(nums[-1], RIGHT, buff=0.09)
        equals_3=Tex("=").scale(0.2).next_to(y3, RIGHT, buff=0.12)
        d_3=Tex("0.00").set_color(GREEN).scale(0.18).next_to(y3, RIGHT, buff=0.28)
        error_equations=VGroup(minus_1, equals_1, d_1, minus_2, equals_2, d_2, minus_3, equals_3, d_3)


        self.play(FadeIn(p34_blocks[1][-1]), FadeOut(dldh_border),
                  dldh_eq.animate.set_color(CHILL_BROWN).scale(0.75).move_to([0.93, 0.58, 0]))
        self.play(Write(minus_1), Write(equals_1), Write(d_1))
        self.wait()

        d_1_copy=d_1.copy()
        d_1_copy.scale(0.85)
        d_1_copy.move_to([1.61, 0.57, 0])

        self.play(ShowCreation(p34_blocks[4]), self.frame.animate.reorient(0, 0, 0, (0.03, -0.01, 0.0), 2.05))
        self.play(ReplacementTransform(d_1.copy(), d_1_copy))
        self.wait()

        d_2_copy=d_2.copy()
        d_2_copy.scale(0.85)
        d_2_copy.move_to([1.66, 0.08, 0])

        self.play(Write(minus_2), Write(equals_2), Write(d_2))
        self.wait()
        self.play(ShowCreation(p34_blocks[5]))
        self.play(ReplacementTransform(d_2.copy(), d_2_copy))
        self.wait()

        self.play(FadeIn(minus_3), FadeIn(equals_3), FadeIn(d_3))
        self.wait()

        #p35
        self.play(rect_1.animate.set_opacity(0.3), rect_2.animate.set_opacity(0.3), 
                  FadeOut(p34_blocks[5]), FadeOut(p34_blocks[4]), FadeOut(d_1_copy), FadeOut(d_2_copy), 
                  self.frame.animate.reorient(0, 0, 0, (-0.17, -0.07, 0.0), 1.81))



        dldmb=Tex(r"\frac{\partial L}{\partial m_2}")
        modular_eq_1b=Tex("=")
        modular_eq_2b=Tex(r"\frac{\partial h_2}{\partial m_2}")
        modular_eq_3b=Tex(r"\cdot")
        modular_eq_4b=Tex(r"\frac{\partial L}{\partial h_2}")

        modular_eqb=VGroup(dldmb, modular_eq_1b, modular_eq_2b, modular_eq_3b, modular_eq_4b)
        dldmb.scale(0.17)
        modular_eq_1b.scale(0.22)
        modular_eq_2b.scale(0.17)
        modular_eq_3b.scale(0.35)
        modular_eq_4b.scale(0.17)
        dldmb.move_to([-1.5, -0.7, 0])
        modular_eq_1b.next_to(dldmb, RIGHT, buff=0.15)
        modular_eq_2b.next_to(modular_eq_1b, RIGHT, buff=0.14)
        modular_eq_3b.next_to(modular_eq_2b, RIGHT, buff=0.35)
        modular_eq_4b.next_to(modular_eq_3b, RIGHT, buff=0.33)

        self.play(FadeIn(modular_eqb))
        self.wait()

        cross_out_line=Line(ORIGIN, RIGHT*0.3).set_stroke(color=YELLOW, width=3)
        cross_out_line.rotate(DEGREES*45)
        cross_out_line.move_to(modular_eq_4b)

        y_minus_yhat=Tex("(\hat{y}-y)").scale(0.18).set_color(YELLOW)
        y_minus_yhat.next_to(modular_eq_4b, RIGHT, buff=0.05)


        self.play(ReplacementTransform(dldh_eq[-4:].copy(), y_minus_yhat[1:-1]), run_time=2.0)
        self.add(y_minus_yhat)
        self.play(ShowCreation(cross_out_line))
        self.wait()

        self.play(rect_1.animate.set_opacity(0.0),rect_2.animate.set_opacity(0.0), 
                  FadeOut(nums[1]), FadeOut(nums[3]), FadeOut(nums[4]), FadeOut(nums[6]),
                  FadeOut(nums[7]), FadeOut(nums[9:]), FadeOut(y1), FadeOut(y2), FadeOut(y3), 
                  FadeOut(error_equations), FadeOut(p34_blocks[1]), FadeOut(dldh_eq))

        self.remove(net_background[:20])

        self.add(net_background)

        self.wait()
        self.embed()







