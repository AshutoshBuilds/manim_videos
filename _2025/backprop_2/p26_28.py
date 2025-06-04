from manimlib import *
import glob

# CHILL_BROWN='#cabba6'
# BLUE = '#00ffff'
# YELLOW = '#ffd35a'
# GREEN = '#00a14b'

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



class p26_28(InteractiveScene):
    def construct(self):
        '''
        
        '''
        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.frame.reorient(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        # self.frame.reorient(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)
        # self.frame.reorient(0, 0, 0, (-0.31, -0.02, 0.0), 1.62)
        self.frame.reorient(0, 0, 0, (-0.56, 0.23, 0.0), 1.56)

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
         
        yhat2 = Tex("0.086").set_color(YELLOW)
        yhat2.scale(0.14)
        yhat2.move_to([1.205, 0.18, 0])
        loss = Tex(format_number(2.45, total_chars=6)).set_color(YELLOW)
        loss.scale(0.16)
        loss.move_to([1.5, 0.18, 0])

        self.add(layers[1], yhat2, loss, layers[3])

        m2_label=net_background[42:44].copy().scale(1.2).move_to([1.48, -0.67, 0])
        loss_label=layers[1][:4].copy().scale(0.75).move_to([0.7, 0.05, 0])
        self.add(m2_label, loss_label)

        m0=nums[2].copy().move_to([0.79, -0.68, 0])
        loss_copy = loss.copy().move_to([0.7, -0.05, 0])
        self.add(layers[2], m0, loss_copy)

        tm2 = Tex(format_number(0.1, total_chars=6)).set_color(YELLOW)
        tm2.scale(0.15)
        tm2.move_to([-1.155, 0.015+0.23, 0])
        self.remove(nums[2])
        self.add(tm2)

        #Ok, now update all yhs and yhats at once!
        w=[1, 0.1, -1, 0, 0, 0]
        logits=[2.34, 0.23, -2.34]
        yhats=[0.885, 0.107, 0.008]
        nums_2=get_numbers_2(x, w, logits, yhats)
        nums_2.shift([0,.23,0])
        for i in [-6, -5, -4, -3, -2, -1]:
            self.remove(nums[i])
            self.add(nums_2[i])

        yhat2_new = nums_2[-2].copy()
        loss_new = Tex(format_number(2.24, total_chars=6)).set_color(YELLOW)
        loss_new.scale(0.16)
        loss_new.move_to([1.5, 0.18, 0])
        self.remove(yhat2)
        yhat2_new.scale(0.9).move_to([1.205, 0.18, 0])
        self.add(yhat2_new)

        self.remove(loss)
        self.add(loss_new)

        m1=nums_2[2].copy().move_to([1.25, -0.68, 0])
        loss_copy_2 = loss_new.copy().move_to([0.7, -0.49, 0])
        self.add(m1, loss_copy_2)

        self.add(layers[4], layers[5], layers[6])
        self.remove(layers[1], yhat2, loss, yhat2_new, loss_new)

        net=VGroup(net_background, nums, nums_2, tm2)
        net.scale(0.8)
        net.move_to([0.2,-0.31,0])
        # nums.set_color(CHILL_BROWN)
        # nums_2.set_color(CHILL_BROWN)
        # tm2.set_color(CHILL_BROWN)

        self.frame.reorient(0, 0, 0, (0.27, -0.28, 0.0), 1.74)
        self.wait()

        yellow_cross=SVGMobject(svg_path+'/yellow_cross.svg')
        yellow_cross.scale(0.25)
        yellow_cross.move_to([1.3, -0.15, 0])

        self.play(ShowCreation(yellow_cross))

        dLdm2=Tex(r"\frac{\partial L}{\partial m_2}")
        dLdm2.set_color(YELLOW)
        dLdm2.scale(0.18)
        dLdm2.move_to([1.3, 0.2, 0])
        # self.add(dLdm2)
        self.play(FadeIn(dLdm2))
        self.wait()



        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(GREEN)
        
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq[9:11].set_color(BLUE)
        softmax_eq[13:15].set_color(YELLOW)
        softmax_eq[17:19].set_color(GREEN)
        
        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2[5:7].set_color(YELLOW)
        softmax_eq_2[9:11].set_color(BLUE)
        softmax_eq_2[13:15].set_color(YELLOW)
        softmax_eq_2[17:19].set_color(GREEN)
        
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")
        
        big_eq = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}})")
        big_eq[10:12].set_color(YELLOW)
        big_eq[14:16].set_color(BLUE)
        big_eq[18:20].set_color(YELLOW)
        big_eq[22:24].set_color(GREEN)
        
        big_eq_2 = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}})")
        big_eq_2[10:12].set_color(YELLOW)
        big_eq_2[14:20].set_color(BLUE)
        big_eq_2[22:24].set_color(YELLOW)
        big_eq_2[26:28].set_color(GREEN)
        
        
        big_eq_3 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}})")
        big_eq_3[10:16].set_color(YELLOW)
        big_eq_3[18:24].set_color(BLUE)
        big_eq_3[26:32].set_color(YELLOW)
        big_eq_3[34:36].set_color(GREEN)
        
        big_eq_4 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}})")
        big_eq_4[10:16].set_color(YELLOW)
        big_eq_4[18:24].set_color(BLUE)
        big_eq_4[26:32].set_color(YELLOW)
        big_eq_4[34:40].set_color(GREEN)
        
        final_eq = Tex(
            r"\frac{\partial Loss}{\partial m_2} = \frac{\partial}{\partial m_2}"
            r"\left[ -\ln\left(\frac{e^{m_2x+b_2}}{e^{m_1x+b_1} + e^{m_2x+b_2} + e^{m_3x+b_3}}\right) \right]"
        )
        final_eq[21:27].set_color(YELLOW)
        final_eq[29:35].set_color(BLUE)
        final_eq[37:43].set_color(YELLOW)
        final_eq[45:51].set_color(GREEN)
        
        h_eqs = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).move_to(([-5.13787109, 0, 0]))
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN)
        center = [(h_eqs.get_right()[0] + RIGHT_SIDE[0]) / 2, 0, 0]
        center_eqs.move_to(center)
        
        center_eqs_2 = VGroup(softmax_eq_2, cross_entropy_loss_eq_2).arrange(DOWN).move_to(center)
        big_eq.move_to(center)
        big_eq_2.move_to(center)
        big_eq_3.move_to(center)
        big_eq_4.move_to(ORIGIN)


        center_eqs.scale(0.16)
        center_eqs.move_to([0.7, -1.0, 0])
        center_eqs_2.scale(0.16)
        center_eqs_2.move_to([0.7, -1.0, 0])
        h_eqs.scale(0.16)
        h_eqs.move_to([-0.2, -1.0, 0])

        self.wait()
        # self.play(self.frame.animate.reorient(0, 0, 0, (0.28, -0.53, 0.0), 1.76))
        temp_opacity=0.5
        self.play(FadeOut(yellow_cross), FadeOut(layers[5]), FadeOut(layers[6]), FadeOut(m0), FadeOut(m1), 
                  FadeOut(loss_copy), FadeOut(loss_copy_2), dLdm2.animate.move_to([1.15, -0.15, 0]).set_opacity(temp_opacity),
                  layers[3].animate.set_opacity(temp_opacity), layers[2].animate.set_opacity(temp_opacity), layers[4].animate.set_opacity(temp_opacity), 
                  loss_label.animate.set_opacity(temp_opacity), m2_label.animate.set_opacity(temp_opacity), tm2.animate.set_opacity(temp_opacity),
                  nums[:2].animate.set_opacity(temp_opacity), nums[3:-6].animate.set_opacity(temp_opacity), nums_2[-7:].animate.set_opacity(temp_opacity),
                  net_background.animate.set_opacity(temp_opacity),
                  self.frame.animate.reorient(0, 0, 0, (0.25, -0.58, 0.0), 1.64))

        self.play(Write(center_eqs), Write(h_eqs), run_time=2)


        ce_label=Text('Cross-Entropy Loss', font='myriad-pro')
        ce_label.set_color(CHILL_BROWN)
        ce_label.scale(0.08)
        ce_label.next_to(cross_entropy_loss_eq, RIGHT)

        softmax_label=Text('Softmax', font='myriad-pro')
        softmax_label.set_color(CHILL_BROWN)
        softmax_label.scale(0.08)
        softmax_label.next_to(softmax_eq, RIGHT)

        self.play(FadeIn(ce_label), FadeIn(softmax_label))
        self.wait()

        box = SurroundingRectangle(cross_entropy_loss_eq, color=YELLOW, buff=0.025)
        self.play(ShowCreation(box))
        self.wait()

        self.play(FadeOut(box))
        self.play(
            ReplacementTransform(center_eqs, center_eqs_2), run_time=2
        )
        self.wait()
        
        big_eq.scale(0.16)
        big_eq.move_to([0.7, -1.0, 0])
        big_eq_2.scale(0.16)
        big_eq_2.move_to([0.7, -1.0, 0])
        big_eq_3.scale(0.16)
        big_eq_3.move_to([0.7, -1.0, 0])
        big_eq_4.scale(0.16)
        big_eq_4.move_to([0.7, -1.0, 0])
        final_eq.scale(0.16)
        final_eq.move_to([0.7, -1.0, 0])

        self.play(
            FadeOut(ce_label), FadeOut(softmax_label),
            ReplacementTransform(cross_entropy_loss_eq_2[0:9], big_eq[0:9]),
            ReplacementTransform(cross_entropy_loss_eq_2[12], big_eq[24]),
            ReplacementTransform(softmax_eq_2[4:19], big_eq[9:24]),
            FadeOut(VGroup(cross_entropy_loss_eq_2[9:12], softmax_eq_2[0:4])), run_time=2
        )
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq[0:9], big_eq_2[0:9]),
            ReplacementTransform(big_eq[24], big_eq_2[28]),
            ReplacementTransform(big_eq[9:14], big_eq_2[9:14]),
            ReplacementTransform(big_eq[15:24], big_eq_2[19:28]),
            ReplacementTransform(h1_eq[3:9], big_eq_2[14:20]),
            FadeOut(h1_eq[0:3]),
            FadeOut(big_eq[14:16]), run_time=2
        )
        self.wait()    

        self.play(
            ReplacementTransform(big_eq_2[0:10], big_eq_3[0:10]),
            ReplacementTransform(big_eq_2[28], big_eq_3[36]),
            ReplacementTransform(big_eq_2[12], big_eq_3[16]),
            ReplacementTransform(big_eq_2[13:22], big_eq_3[17:26]),
            ReplacementTransform(big_eq_2[24:28], big_eq_3[32:36]),
            FadeOut(VGroup(big_eq_2[10:12], big_eq_2[22:24], h2_eq[0:3])),
            ReplacementTransform(h2_eq[3:9].copy(), big_eq_3[10:16]),
            ReplacementTransform(h2_eq[3:9], big_eq_3[26:32]), run_time=2
        )
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq_3[0:35], big_eq_4[0:35]),
            ReplacementTransform(big_eq_3[36], big_eq_4[40]),
            FadeOut(big_eq_3[34:36]),
            ReplacementTransform(h3_eq[3:9], big_eq_4[34:40]),
            FadeOut(h3_eq[0:3]), run_time=2
        )
        
        self.wait()
        # self.play(big_eq_4.animate.shift(UP * 2), run_time=2)
            
        final_eq.move_to([ 0.7, -1.22 ,  0. ])    
        # self.add(final_eq)

        self.play(ReplacementTransform(dLdm2[0].copy(), final_eq[0]),
                  ReplacementTransform(dLdm2[2:].copy(), final_eq[5:9]), 
                  ReplacementTransform(big_eq_4[0:4].copy(), final_eq[1:5]),
                  run_time=1.5)
        self.add(final_eq[9]) #Equals sign
        self.play(ReplacementTransform(dLdm2[0].copy(), final_eq[10]),
                  ReplacementTransform(dLdm2[2:].copy(), final_eq[11:15]),
                  run_time=1.5)
        self.play(ReplacementTransform(big_eq_4[5:41].copy(), final_eq[16:52]))
        # self.play(Write(final_eq[52]), Write(final_eq[15]))
        self.add(final_eq[52], final_eq[15])
        self.wait()
        
        big_derivative=SVGMobject(svg_path+'/big_derivative.svg')
        big_derivative.scale(0.75)
        big_derivative.move_to([1.35, -2.15, 0])

        self.play(FadeIn(big_derivative), 
                  self.frame.animate.reorient(0, 0, 0, (0.81, -1.9, 0.0), 2.22), run_time=2.0)
        self.wait()

        #Cool now move back to big newtork!
        temp_opacity=0.8
        temp_opacity_2=0.5 #Use to make numbers more chill
        self.play(self.frame.animate.reorient(0, 0, 0, (0.21, -0.42, 0.0), 1.71), 
                  FadeOut(big_derivative), FadeOut(final_eq), FadeOut(big_eq_4),
                  dLdm2.animate.set_opacity(temp_opacity),
                  layers[3].animate.set_opacity(temp_opacity), layers[2].animate.set_opacity(temp_opacity), layers[4].animate.set_opacity(temp_opacity), 
                  loss_label.animate.set_opacity(temp_opacity), m2_label.animate.set_opacity(temp_opacity), tm2.animate.set_opacity(temp_opacity_2),
                  nums[:2].animate.set_opacity(temp_opacity_2), nums[3:-6].animate.set_opacity(temp_opacity_2), nums_2[-7:].animate.set_opacity(temp_opacity_2),
                  net_background.animate.set_opacity(temp_opacity), run_time=3.0)

        self.wait()


        #Ok lets keep rolling to p29 here - first I need some opaque rounded rectangles. 
        rect1=RoundedRectangle(0.28, 0.48, 0.02)
        rect1.set_stroke(width=0)
        rect1.set_color(CHILL_GREEN)
        rect1.set_opacity(0.4)

        rect2=RoundedRectangle(0.225, 0.48, 0.02)
        rect2.set_stroke(width=0)
        rect2.set_color(CHILL_BLUE)
        rect2.set_opacity(0.4)

        rect1.move_to([-0.19, -0.36,  1])
        rect2.move_to([0.142, -0.36,  1])

        dLdm2b=dLdm2.copy()
        dLdm2b.set_color(WHITE).set_opacity(1.0)
        dLdm2b.scale(0.9)
        dLdm2b.move_to([-1.0, -0.9, 0])

        modular_eq_1=Tex("=")
        modular_eq_2=Tex(r"\frac{\partial h}{\partial m_2}")
        modular_eq_3=Tex(r"\cdot")
        modular_eq_4=Tex(r"\frac{\partial L}{\partial h}")

        modular_eq=VGroup(modular_eq_1, modular_eq_2, modular_eq_3, modular_eq_4)
        modular_eq_1.scale(0.17)
        modular_eq_2.scale(0.17)
        modular_eq_3.scale(0.35)
        modular_eq_4.scale(0.17)
        modular_eq_1.move_to([-0.85, -0.9, 0])
        modular_eq_2.move_to([-0.55, -0.9, 0])
        modular_eq_3.move_to([-0.2, -0.9, 0])
        modular_eq_4.move_to([0.1, -0.9, 0])


        self.wait()
        self.play(FadeIn(rect1))
        self.play(FadeIn(rect2))
        self.wait()
        self.play(FadeIn(modular_eq_2))
        self.play(FadeIn(modular_eq_4))
        self.wait()
        self.play(ReplacementTransform(dLdm2, dLdm2b), run_time=3.0)
        self.add(modular_eq_1, modular_eq_3)
        self.wait()

        # self.add(rect1, rect2, dLdm2b, modular_eq)


        #Starting p30
        blocks=VGroup()
        for p in sorted(glob.glob(svg_path+'/p30_31/*.svg')):
            blocks.add(SVGMobject(p)[1:])  


        blocks[0].scale(0.9)
        blocks[0].move_to([0.3, -0.4, 0])

        rect3=RoundedRectangle(0.5, 0.5, 0.02)
        rect3.set_stroke(width=0)
        rect3.set_color(CHILL_GREEN)
        rect3.set_opacity(0.4)
        rect3.move_to([-0.13, -0.4, 0])

        rect4=RoundedRectangle(0.5, 0.5, 0.02)
        rect4.set_stroke(width=0)
        rect4.set_color(CHILL_BLUE)
        rect4.set_opacity(0.4)
        rect4.move_to([0.725, -0.4, 0])

        temp_opacity=0.0
        self.wait()
        self.play(FadeOut(modular_eq), FadeOut(dLdm2b),
                  dLdm2.animate.set_opacity(temp_opacity),
                  layers[3].animate.set_opacity(temp_opacity), layers[2].animate.set_opacity(temp_opacity), layers[4].animate.set_opacity(temp_opacity), 
                  loss_label.animate.set_opacity(temp_opacity), m2_label.animate.set_opacity(temp_opacity), tm2.animate.set_opacity(temp_opacity),
                  nums[:2].animate.set_opacity(temp_opacity), nums[3:-6].animate.set_opacity(temp_opacity), nums_2[-7:].animate.set_opacity(temp_opacity),
                  net_background.animate.set_opacity(temp_opacity), run_time=1.2)

        self.play(ReplacementTransform(rect1, rect3), 
                 ReplacementTransform(rect2, rect4), run_time=3.0)
        self.add(blocks[0])
        self.wait()

        simple_eq_1=Tex("y=2x")
        simple_eq_2=Tex("z=4y")

        simple_eq_1.scale(0.22)
        simple_eq_1.move_to(rect3.get_center())
        simple_eq_2.scale(0.22)
        simple_eq_2.move_to(rect4.get_center())

        # self.add(simple_eq_1, simple_eq_2)

        self.play(Write(simple_eq_1))
        self.wait()
        self.play(Write(simple_eq_2))
        self.wait()


        simple_eq_3=Tex("z=4y=4(2x)=8x")
        simple_eq_3.scale(0.22)
        simple_eq_3.move_to([0.3, -0.9, 0])

        self.play(ReplacementTransform(simple_eq_2.copy(), simple_eq_3[:4]))
        self.add(simple_eq_3[4])
        self.wait()
        self.add(simple_eq_3[5:7])
        self.play(ReplacementTransform(simple_eq_1[2:].copy(), simple_eq_3[7:9]))
        self.add(simple_eq_3[9])
        self.wait()
        self.play(FadeIn(simple_eq_3[10:]))
        self.wait()

        blocks[1].scale(0.9)
        blocks[1].move_to([0.36, -1.32, 0])

        self.play(self.frame.animate.reorient(0, 0, 0, (0.21, -0.83, 0.0), 1.71),
                    ShowCreation(blocks[1]), run_time=1.5)

        blocks[2].scale(0.9)
        blocks[2].move_to([0.30, -0.35, 0])

        dydx=Tex(r"\frac{d y}{d x}=2")
        dydx.scale(0.15)
        dydx.move_to([-0.10, -0.25, 0])

        dzdy=Tex(r"\frac{d z}{d y}=4")
        dzdy.scale(0.15)
        dzdy.move_to([0.85, -0.35, 0])

        self.wait()
        self.play(FadeOut(blocks[1]), FadeOut(simple_eq_3), FadeIn(blocks[2]),
                  simple_eq_1.animate.move_to(rect3.get_center()+np.array([0,-0.17,0])), 
                  simple_eq_2.animate.move_to(rect4.get_center()+np.array([0,-0.17,0])))


        self.play(Write(dydx))
        self.wait()
        self.play(Write(dzdy))




        self.add(blocks[2])



        # self.add(simple_eq_3)


        self.wait()









        # self.add(big_derivative)
        # self.frame.reorient(0, 0, 0, (0.76, -1.85, 0.0), 2.21)
        # self.play(ShowCreation(big_derivative))


        # self.play(
        #     Write(final_eq[0]),
        #     Write(final_eq[5:16]),
        #     ReplacementTransform(big_eq_4[0:4], final_eq[1:5]),
        #     ReplacementTransform(big_eq_4[4], final_eq[9]),
        #     Write(final_eq[52]),
        #     ReplacementTransform(big_eq_4[5:41].copy(), final_eq[16:52]), run_time=2
        # )

        #Ok so now fade out network and stuff, move camera down, and substitute. 
        #Probably fade and camera move first. 
        #Ok as I'm hacking through here I'm realizing that I don't want ot fully lose the context of 
        # the nerual network diagram


        # self.play(FadeOut(yellow_cross), FadeOut(layers[5]), FadeOut(layers[6]), FadeOut(m0), FadeOut(m1), 
        #           FadeOut(loss_copy), FadeOut(loss_copy_2), dLdm2.animate.move_to([1.15, -0.15, 0]).set_opacity(0.3),
        #           layers[3].animate.set_opacity(0.2), layers[2].animate.set_opacity(0.2), layers[4].animate.set_opacity(0.2), 
        #           loss_label.animate.set_opacity(0.2), m2_label.animate.set_opacity(0.2), tm2.animate.set_opacity(0.2),
        #           nums[:2].animate.set_opacity(0.2), nums[3:-6].animate.set_opacity(0.2), nums_2[-7:].animate.set_opacity(0.2),
        #           net_background.animate.set_opacity(0.2),
        #           self.frame.animate.reorient(0, 0, 0, (0.47, -1.09, 0.0), 1.71), run_time=1.5)

        # dLdm2.move_to([1.15, -0.15, 0]).set_opacity(0.2)
  

        # self.play(FadeOut(net), FadeOut(dLdm2), FadeOut(yellow_cross), FadeOut(layers[1])) 
    
        # self.play(*[o.animate.set_opacity(0.2) for o in [net, dLdm2, yellow_cross, layers[1], yhat2, loss, layers[3], m2_label, loss_label, layers[2], m0, loss_copy]])
        # for o in [net, dLdm2, yellow_cross, layers, yhat2, loss, m2_label, 
        #         loss_label, m0, loss_copy, m1, loss_copy_2]:
        #     self.remove(o)
        # self.play(self.frame.animate.reorient(0, 0, 0, (0.47, -1.09, 0.0), 1.71))
        # self.wait()
        







        self.wait()






class P27V3Slow(InteractiveScene):
    def construct(self):
        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(GREEN)
        
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq[9:11].set_color(BLUE)
        softmax_eq[13:15].set_color(YELLOW)
        softmax_eq[17:19].set_color(GREEN)
        
        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2[5:7].set_color(YELLOW)
        softmax_eq_2[9:11].set_color(BLUE)
        softmax_eq_2[13:15].set_color(YELLOW)
        softmax_eq_2[17:19].set_color(GREEN)
        
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")
        
        big_eq = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}})")
        big_eq[10:12].set_color(YELLOW)
        big_eq[14:16].set_color(BLUE)
        big_eq[18:20].set_color(YELLOW)
        big_eq[22:24].set_color(GREEN)
        
        big_eq_2 = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}})")
        big_eq_2[10:12].set_color(YELLOW)
        big_eq_2[14:20].set_color(BLUE)
        big_eq_2[22:24].set_color(YELLOW)
        big_eq_2[26:28].set_color(GREEN)
        
        
        big_eq_3 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}})")
        big_eq_3[10:16].set_color(YELLOW)
        big_eq_3[18:24].set_color(BLUE)
        big_eq_3[26:32].set_color(YELLOW)
        big_eq_3[34:36].set_color(GREEN)
        
        big_eq_4 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}})")
        big_eq_4[10:16].set_color(YELLOW)
        big_eq_4[18:24].set_color(BLUE)
        big_eq_4[26:32].set_color(YELLOW)
        big_eq_4[34:40].set_color(GREEN)
        
        final_eq = Tex(
            r"\frac{\partial Loss}{\partial m_2} = \frac{\partial}{\partial m_2}"
            r"\left[ -\ln\left(\frac{e^{m_2x+b_2}}{e^{m_1x+b_1} + e^{m_2x+b_2} + e^{m_3x+b_3}}\right) \right]"
        )
        final_eq[21:27].set_color(YELLOW)
        final_eq[29:35].set_color(BLUE)
        final_eq[37:43].set_color(YELLOW)
        final_eq[45:51].set_color(GREEN)
        
        h_eqs = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).move_to(([-5.13787109, 0, 0]))
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN)
        
        center = [(h_eqs.get_right()[0] + RIGHT_SIDE[0]) / 2, 0, 0]
        
        center_eqs.move_to(center)
        
        center_eqs_2 = VGroup(softmax_eq_2, cross_entropy_loss_eq_2).arrange(DOWN).move_to(center)
        big_eq.move_to(center)
        big_eq_2.move_to(center)
        big_eq_3.move_to(center)
        big_eq_4.move_to(ORIGIN)
        
            
        self.play(Write(center_eqs), Write(h_eqs), run_time=2)
        
        self.wait()
        
        self.play(
            ReplacementTransform(center_eqs, center_eqs_2), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(cross_entropy_loss_eq_2[0:9], big_eq[0:9]),
            ReplacementTransform(cross_entropy_loss_eq_2[12], big_eq[24]),
            ReplacementTransform(softmax_eq_2[4:19], big_eq[9:24]),
            FadeOut(VGroup(cross_entropy_loss_eq_2[9:12], softmax_eq_2[0:4])), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq[0:9], big_eq_2[0:9]),
            ReplacementTransform(big_eq[24], big_eq_2[28]),
            ReplacementTransform(big_eq[9:14], big_eq_2[9:14]),
            ReplacementTransform(big_eq[15:24], big_eq_2[19:28]),
            ReplacementTransform(h1_eq[3:9], big_eq_2[14:20]),
            FadeOut(h1_eq[0:3]),
            FadeOut(big_eq[14:16]), run_time=2
        )
            
        self.wait()    
                                
        self.play(
            ReplacementTransform(big_eq_2[0:10], big_eq_3[0:10]),
            ReplacementTransform(big_eq_2[28], big_eq_3[36]),
            ReplacementTransform(big_eq_2[12], big_eq_3[16]),
            ReplacementTransform(big_eq_2[13:22], big_eq_3[17:26]),
            ReplacementTransform(big_eq_2[24:28], big_eq_3[32:36]),
            FadeOut(VGroup(big_eq_2[10:12], big_eq_2[22:24], h2_eq[0:3])),
            ReplacementTransform(h2_eq[3:9].copy(), big_eq_3[10:16]),
            ReplacementTransform(h2_eq[3:9], big_eq_3[26:32]), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq_3[0:35], big_eq_4[0:35]),
            ReplacementTransform(big_eq_3[36], big_eq_4[40]),
            FadeOut(big_eq_3[34:36]),
            ReplacementTransform(h3_eq[3:9], big_eq_4[34:40]),
            FadeOut(h3_eq[0:3]), run_time=2
        )
        
        self.wait()
        
        self.play(big_eq_4.animate.shift(UP * 2), run_time=2)
        
        self.wait()
        
        self.play(
            Write(final_eq[0]),
            Write(final_eq[5:16]),
            ReplacementTransform(big_eq_4[0:4], final_eq[1:5]),
            ReplacementTransform(big_eq_4[4], final_eq[9]),
            Write(final_eq[52]),
            ReplacementTransform(big_eq_4[5:41].copy(), final_eq[16:52]), run_time=2
        )