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


class p40_44(InteractiveScene):
    def construct(self):
        '''
        
        '''
        # net_background=SVGMobject(svg_path+'/p44_background_1.svg')[1:]
        net_background=SVGMobject(svg_path+'/p40_net_background_manim.svg')[1:]

        
        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.35, 0, -2.35]
        yhats=[0.91, 0.09, 0.00]
        nums=get_numbers_3(x, w, logits, yhats)
    
        m1_label=Tex('m_1').set_color(CHILL_BROWN).scale(0.11)
        m1_label.move_to([-1.19, 0.14, 0])
        b1_label=Tex('b_1').set_color(CHILL_BROWN).scale(0.11)
        b1_label.move_to([-0.87, 0.29, 0])
        m2_label=Tex('m_2').set_color(CHILL_BROWN).scale(0.11)
        m2_label.move_to([-1.155, -0.05, 0])
        b2_label=Tex('b_2').set_color(CHILL_BROWN).scale(0.11)
        b2_label.move_to([-0.87, -0.055, 0])
        m3_label=Tex('m_3').set_color(CHILL_BROWN).scale(0.11)
        m3_label.move_to([-1.16, -0.23, 0])
        b3_label=Tex('b_3').set_color(CHILL_BROWN).scale(0.11)
        b3_label.move_to([-0.87, -0.4, 0])
        manual_labels=VGroup(m1_label, b1_label, m2_label, b2_label, m3_label, b3_label)
        # self.add(net_background, nums, manual_labels)

        full_gradient=Tex(r"\left[ \frac{\partial L}{\partial m_1}, \hspace{1mm} \frac{\partial L}{\partial m_2}, \hspace{1mm} \frac{\partial L}{\partial m_3}, \hspace{1mm} \frac{\partial L}{\partial b_1}, \hspace{1mm} \frac{\partial L}{\partial b_2}, \hspace{1mm} \frac{\partial L}{\partial b_3} \right]")
        full_gradient.scale(0.2)
        self.add(full_gradient)
        self.frame.reorient(0, 0, 0, (0.0, 0.0, 0.0), 1.75)

        grade_eq_scale=0.15
        grad_1=Tex(r"\frac{\partial L}{\partial m_1} = x(\hat{y}_1-y_1)=2.14").scale(grade_eq_scale)
        grad_2=Tex(r"\frac{\partial L}{\partial m_2} = x(\hat{y}_2-y_2)=-2.14 ").scale(grade_eq_scale)
        grad_3=Tex(r"\frac{\partial L}{\partial m_3} = x(\hat{y}_3-y_3)=0.00 ").scale(grade_eq_scale)
        grad_4=Tex(r"\frac{\partial L}{\partial b_1} = (\hat{y}_1-y_1)=0.91").scale(grade_eq_scale)
        grad_5=Tex(r"\frac{\partial L}{\partial b_2} = (\hat{y}_2-y_2)=-0.91").scale(grade_eq_scale)
        grad_6=Tex(r"\frac{\partial L}{\partial b_3} = (\hat{y}_3-y_3)=0.00" ).scale(grade_eq_scale)
        grad_eqs=VGroup(grad_1, grad_2, grad_3, grad_4, grad_5, grad_6)
        grad_1[-4:].set_color('#00FFFF')
        grad_2[-5:].set_color(YELLOW)
        grad_3[-4:].set_color(GREEN)
        grad_4[-4:].set_color('#00FFFF')
        grad_5[-5:].set_color(YELLOW)
        grad_6[-4:].set_color(GREEN)

        vertical_spacing=0.2
        x_start=0.55
        for i, g in enumerate(grad_eqs):
            g.move_to([x_start, vertical_spacing*(2.5-i), 0], aligned_edge=LEFT)

        self.wait()

        self.play(ReplacementTransform(full_gradient[1:7], grad_1[:6]),
                  ReplacementTransform(full_gradient[8:14], grad_2[:6]),
                  ReplacementTransform(full_gradient[15:21], grad_3[:6]),
                  ReplacementTransform(full_gradient[22:28], grad_4[:6]),
                  ReplacementTransform(full_gradient[29:35], grad_5[:6]),
                  ReplacementTransform(full_gradient[36:42], grad_6[:6]),
                  FadeOut(full_gradient[0]), 
                  FadeOut(full_gradient[7]), 
                  FadeOut(full_gradient[14]), 
                  FadeOut(full_gradient[21]), 
                  FadeOut(full_gradient[28]), 
                  FadeOut(full_gradient[35]),
                  FadeOut(full_gradient[42]),
                  FadeIn(net_background), 
                  FadeIn(manual_labels),
                  FadeIn(nums),
                  self.frame.animate.reorient(0, 0, 0, (-0.1, -0.02, 0.0), 1.94),
                  run_time=2.4
                  )

        self.wait()
        self.add(grad_2[6:-6])
        self.wait()
        self.play(Write(grad_1[6:-5]),
                  Write(grad_3[6:-5]),
                  Write(grad_4[6:-5]),
                  Write(grad_5[6:-7]),
                  Write(grad_6[6:-5]),
             )
        self.wait()


        
        vector_eqs=SVGMobject(svg_path+'/vector_eqs_1.svg')
        vector_eqs.scale(0.58)
        vector_eqs.move_to([1.4,0,0])

        self.play(ShowCreation(vector_eqs), 
                 self.frame.animate.reorient(0, 0, 0, (0.01, -0.04, 0.0), 1.96))
        self.wait()

        self.play(FadeOut(vector_eqs))
        self.play(Write(grad_1[-5:]),
                Write(grad_2[-6:]),
                Write(grad_3[-5:]),
                Write(grad_4[-5:]),
                Write(grad_5[-6:]),
                Write(grad_6[-5:]))
        

        p42_blocks=VGroup()
        for p in sorted(glob.glob(svg_path+'/p42/*.svg')):
            p42_blocks.add(SVGMobject(p)[1:])    
        p42_blocks.move_to([-0.01, 0.13, 0])
        self.wait()

        grad_2_val=grad_2[-5:].copy()
        self.play(FadeIn(p42_blocks[1]), self.frame.animate.reorient(0, 0, 0, (-0.12, 0.23, 0.0), 1.94), run_time=2.0)
        self.play(ShowCreation(p42_blocks[3]), grad_2_val.animate.move_to([0.15, 0.85, 0]), run_time=1.5)
        self.wait()

        box=SurroundingRectangle(nums[-2], color=YELLOW, buff=0.025)
        self.play(ShowCreation(box))
        self.wait()

        w=[1, 0.1, -1, 0, 0, 0]
        logits=[2.35, 0.24, -2.35]
        yhats=[0.89, 0.11, 0.00]
        nums2=get_numbers_3(x, w, logits, yhats)

        box2=SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        self.play(ShowCreation(box2))
        self.wait()

        self.play(ReplacementTransform(nums[2], nums2[2]))
        self.play(ReplacementTransform(nums[8], nums2[8]),
                  ReplacementTransform(nums[11], nums2[11]), 
                  ReplacementTransform(nums[10], nums2[10]),
                  ReplacementTransform(nums[12], nums2[12]))
        self.wait()
        self.play(FadeOut(box), FadeOut(box2))
        self.wait()

        box3=SurroundingRectangle(grad_1[-4:], color="#00FFFF", buff=0.025)
        self.play(ShowCreation(box3))
        self.wait()

        grad_1_val=grad_1[-4:].copy()
        self.play(FadeIn(p42_blocks[0]))
        self.play(ShowCreation(p42_blocks[2]), grad_1_val.animate.move_to([-0.45, 0.855, 0]), run_time=1.5)
        self.wait()

        w=[1.1, 0.1, -1, 0, 0, 0]
        logits=[2.59, 0.24, -2.35]
        yhats=[0.91, 0.09, 0.00]
        nums3=get_numbers_3(x, w, logits, yhats)

        box4=SurroundingRectangle(nums[1], color="#00FFFF", buff=0.025)
        self.play(ShowCreation(box4))

        self.remove(nums[1], nums[7], nums[10:12])
        self.add(nums2[1], nums2[7], nums2[10:12])
        self.wait()
        self.play(ReplacementTransform(nums2[1], nums3[1]))
        self.play(ReplacementTransform(nums2[7], nums3[7]),
                  ReplacementTransform(nums2[11], nums3[11]), 
                  ReplacementTransform(nums2[10], nums3[10]),
                  ReplacementTransform(nums2[12], nums3[12]))
        self.wait()
        self.play(FadeOut(box3), FadeOut(box4))
        self.wait()

        self.play(ShowCreation(p42_blocks[4]))
        self.wait()

        self.play(ShowCreation(p42_blocks[5]))
        self.wait()       
        # Hmm yeah showing all the numbers ticking around as I change m2 would be pretty cool
        # Same thing I do when visualizing training right?
        # Hmm actually I think the numbers ticking right now is going to be ditracting - > but I do 
        # think changing the number once will make sense/be helpful - easier to animate too - let's try that. 

        #Ok time to gear up to move to big gradient descent equation
        #Jump to end and then reconstruct here. Actually let me sort out my fade outs real quick first. 
        # Ok so i probably want to make vectors including brackets in manim, and bring in other stuff from illustraotr
        p43_blocks=VGroup()
        for p in sorted(glob.glob(svg_path+'/p43/*.svg')):
            p43_blocks.add(SVGMobject(p)[1:])    
        p43_blocks.move_to([-0.01, 0.13, 0])





        self.play(FadeOut(net_background),
                  FadeOut(nums[0]),
                  FadeOut(nums3[1]),
                  FadeOut(nums2[2]),
                  FadeOut(nums[3]),
                  FadeOut(nums[4:7]),
                  FadeOut(nums3[7]),
                  FadeOut(nums2[8]),
                  FadeOut(nums[9]),
                  FadeOut(nums3[10]),
                  FadeOut(nums3[11]),
                  FadeOut(nums3[12]),
                  FadeOut(p42_blocks),
                  FadeOut(grad_1_val),
                  FadeOut(grad_2_val),
            )

        self.wait()


        #I think rearrange param vector first, then fade out all the grad math. 




        self.play(FadeOut(grad_1[6:]),
                  FadeOut(grad_2[6:]),
                  FadeOut(grad_3[6:]),
                  FadeOut(grad_4[6:]),
                  FadeOut(grad_5[6:]),
                  FadeOut(grad_6[6:]))



        self.add(p42_blocks)
        








        self.wait(20)
        self.embed()

