from manimlib import *
from functools import partial
# from manimlib.mobject.svg.old_tex_mobject import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

def get_decision_boundary_value(x, w0=1, w1=1, b=1):
    return -(w0/w1)*x - b/w1

def get_region_points(axes, w0=1, w1=1, b=1, above=True):
    # Get points along the decision boundary
    xs=np.linspace(-1.2, 1.2, 100)
    ys=np.array([get_decision_boundary_value(x, w0=w0, w1=w1, b=b) for x in xs])
    ys=np.clip(ys, -1.2, 1.2)

    line_points = [axes.c2p(x, y) for x,y in zip(xs,ys)]
    if above:
        # For region above line: add top corners
        points = line_points + [
            axes.c2p(1.2, 1.2),   # Top right
            axes.c2p(-1.2, 1.2),  # Top left
        ]
    else:
        # For region below line: add bottom corners
        points = line_points + [
            axes.c2p(1.2, -1.2),   # Bottom right
            axes.c2p(-1.2, -1.2),  # Bottom left
        ]
    
    return points

class P45_48v2(InteractiveScene):
    def construct(self):

        axes = Axes(
            x_range=[-1.2, 1.2, 1.0],
            y_range=[-1.2, 1.2, 1.0],
            axis_config={
                # "big_tick_numbers":[-1,1],
                "include_ticks":False,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip":True,
                "tip_config": {  # Changed from tip_shape
                    "fill_opacity": 1,
                    "width": 0.1,
                    "length": 0.1
                }
            },
            # x_axis_config={"include_numbers":True, 
            #                #"big_tick_spacing":0.5
            #                "decimal_number_config":{"num_decimal_places":1, "font_size":30}},

            # x_axis_config={"numbers_with_elongated_ticks": [-1, 1]}
        )
        axes.scale(1.5)
        axes.move_to(LEFT*3+UP)  # Moves it 2 units to the right
        # axes.add_coordinate_labels()

        x_label=Tex('x_0', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes, RIGHT)
        x_label.shift(0.2*DOWN+0.15*LEFT)
        y_label=Tex('x_1', font_size=28).set_color(CHILL_BROWN)
        y_label.next_to(axes, TOP)
        y_label.shift(0.98*DOWN+0.15*RIGHT)

        #Manually add ticks, can't quite figure out how to make them like i want
        t1=Line(axes.c2p(0,1), axes.c2p(0.1,1), stroke_width=1.5)
        t1.set_color(CHILL_BROWN)
        t1l=Tex('1', font_size=24).set_color(CHILL_BROWN)
        t1l.next_to(t1, LEFT*0.4)

        t2=Line(axes.c2p(1,0), axes.c2p(1,0.1), stroke_width=1.5)
        t2.set_color(CHILL_BROWN)
        t2l=Tex('1', font_size=24).set_color(CHILL_BROWN)
        t2l.next_to(t2, DOWN*0.4)

        t3=Line(axes.c2p(-1,0), axes.c2p(-1,0.1), stroke_width=1.5)
        t3.set_color(CHILL_BROWN)
        t3l=Tex('-1', font_size=24).set_color(CHILL_BROWN)
        t3l.next_to(t3, DOWN*0.4)

        t4=Line(axes.c2p(0,-1), axes.c2p(0.1,-1), stroke_width=1.5)
        t4.set_color(CHILL_BROWN)
        t4l=Tex('-1', font_size=24).set_color(CHILL_BROWN)
        t4l.next_to(t4, LEFT*0.4)

        self.add(axes, x_label, y_label, t1, t1l, t2, t2l, t3, t3l, t4, t4l)

        dots = VGroup()
        dots.add(Dot(axes.c2p(-1, -1), radius=0.06).set_color(BLUE))
        dots.add(Dot(axes.c2p(-1, 1), radius=0.06).set_color(YELLOW))
        dots.add(Dot(axes.c2p(1, -1), radius=0.06).set_color(YELLOW))
        dots.add(Dot(axes.c2p(1, 1), radius=0.06).set_color(YELLOW))

        dots_labels=VGroup()
        dots_labels.add(Text('-').set_color(BLUE).move_to(axes.c2p(-0.85, -0.9)))
        dots_labels.add(Tex('+').set_color(YELLOW).move_to(axes.c2p(-0.85, 1.1)))
        dots_labels.add(Tex('+').set_color(YELLOW).move_to(axes.c2p(1.15, -0.9)))
        dots_labels.add(Tex('+').set_color(YELLOW).move_to(axes.c2p(1.15, 1.1)))

        self.wait()
        self.add(dots)     
        self.play(FadeIn(dots), run_time=1)

        self.add(dots_labels)
        self.wait()

        eq1=Tex("\hat{y}=x_0 w_0 + x_1 w_1 + b")
        eq1.set_color(WHITE).scale(0.9)
        eq1.move_to(3*RIGHT+2*UP)
        self.add(eq1)
        self.wait()

        label_box=Square(0.8)
        label_box.set_color(BLUE).move_to(axes.c2p(-1, -1.0))
        label_t1=Tex("x_0=-1", font_size=36)
        label_t1.set_color(BLUE).move_to(axes.c2p(-1.9, -0.75)) #-1.5
        label_t2=Tex("x_1=-1", font_size=36)
        label_t2.set_color(BLUE).move_to(axes.c2p(-1.9, -1.00))    
        label_t3=Tex("y=-1", font_size=36)
        label_t3.set_color(BLUE).move_to(axes.c2p(-1.9, -1.25)) 
        self.add(label_box, label_t1, label_t2)
        self.wait()

        #P46
        self.add(label_t3)
        self.wait()

        #Now add dials
        w0, w1, b = -1, 1, 1

        dial_ticks_1 = ImageMobject('/Users/stephen/manim_videos/_2025/perceptron/single_dial_ticks_brown.png')
        dial_ticks_1.scale(0.45)
        dial_ticks_1.move_to(DOWN*2.5+LEFT*4.85)
        # w0_label=Tex('w_0='+str(round(w0,1)), color=WHITE)
        w0_label=Tex('w_0=', color=WHITE)
        w0_label.scale(0.7).next_to(dial_ticks_1, 0.01*DOWN)
        w0_label.shift(0.35*LEFT)
        w0_value=Tex(f'{w0:.1f}', color=WHITE)
        w0_value.scale(0.7).next_to(w0_label, RIGHT, aligned_edge=BOTTOM)
        w0_value.shift(0.1*LEFT+0.05*UP)


        dial_1=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_1.scale(0.32)
        dial_1.move_to(dial_ticks_1)
        dial_1.shift(0.075*DOWN+0.01*RIGHT)
        dial_1.rotate(-1*w0*60*DEGREES)
        self.add(dial_ticks_1, dial_1, w0_label, w0_value)
        self.wait()


        dial_ticks_2=dial_ticks_1.copy()
        dial_ticks_2.shift(1.8*RIGHT)
        # w1_label=Tex('w_1='+str(round(w1,1)), color=WHITE)
        # w1_label.scale(0.7).next_to(dial_ticks_2, 0.01*DOWN)
        w1_label=Tex('w_1=', color=WHITE)
        w1_label.scale(0.7).next_to(dial_ticks_2, 0.01*DOWN)
        w1_label.shift(0.3*LEFT)
        w1_value=Tex(f'{w1:.1f}', color=WHITE)
        w1_value.scale(0.7).next_to(w1_label, RIGHT, aligned_edge=BOTTOM)
        w1_value.shift(0.1*LEFT+0.05*UP)

        dial_2=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_2.scale(0.32)
        dial_2.move_to(dial_ticks_2)
        dial_2.shift(0.075*DOWN+0.01*RIGHT)
        dial_2.rotate(-1*w1*60*DEGREES)
        self.add(dial_ticks_2, dial_2, w1_label, w1_value)
        self.wait()

        dial_ticks_3=dial_ticks_2.copy()
        dial_ticks_3.shift(1.8*RIGHT)
        # b_label=Tex('b='+str(round(b,1)), color=WHITE)
        # b_label.scale(0.7).next_to(dial_ticks_3, 0.01*DOWN)
        b_label=Tex('b=', color=WHITE)
        b_label.scale(0.7).next_to(dial_ticks_3, 0.01*DOWN)
        b_label.shift(0.3*LEFT+0.08*UP) #Not sure why I have to shift this one specially
        b_value=Tex(f'{b:.1f}', color=WHITE)
        b_value.scale(0.7).next_to(b_label, RIGHT, aligned_edge=BOTTOM)
        b_value.shift(0.1*LEFT) #0.05*UP)
        dial_3=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_3.scale(0.32)
        dial_3.move_to(dial_ticks_3)
        dial_3.shift(0.075*DOWN+0.01*RIGHT)
        dial_3.rotate(-1*b*60*DEGREES)
        self.add(dial_ticks_3, dial_3, b_label, b_value)
        self.wait()

        #Now plug into equation
        eq2=Tex("=(-1)(-1) + (-1)(1) + 1")
        eq2.set_color(WHITE).scale(0.9)
        eq2.next_to(eq1, 0.8*DOWN)
        eq2.shift(0.85*RIGHT)
        self.add(eq2)
        self.wait()

        eq3=Tex("=1")
        eq3.set_color(WHITE).scale(0.9)
        eq3.next_to(eq2, 0.8*DOWN, aligned_edge=LEFT)
        # eq3.shift(0.85*RIGHT)
        self.add(eq3)
        self.wait()


        eq4=Tex("Error=y-\hat{y}")
        eq4.set_color(WHITE).scale(0.9)
        eq4.next_to(eq1, DOWN, aligned_edge=LEFT)
        eq4.shift(1.8*DOWN)
        self.add(eq4)
        self.wait()


        eq5=Tex("=-1-1=-2")
        eq5.set_color(WHITE).scale(0.9)
        eq5.next_to(eq4, DOWN, aligned_edge=LEFT)
        eq5.shift(1.3*RIGHT)
        self.add(eq5)
        self.wait()

        eq6=Tex("Error^2=(y-\hat{y})^2=4")
        eq6.set_color(WHITE).scale(0.9)
        eq6.next_to(eq1, DOWN, aligned_edge=LEFT)
        eq6.shift(3*DOWN)
        self.add(eq6)
        self.wait()


        #P47
        self.play(self.frame.animate.reorient(0, 0, 0, (-3.0, -2.24, 0.0), 4.10))
        self.wait()


        #dial_1.rotate(-1*w0*60*DEGREES)
        w0=-0.9
        w0_value_b=Tex(f'{w0:.1f}', color=WHITE)
        w0_value_b.scale(0.7).next_to(w0_label, RIGHT, aligned_edge=BOTTOM)
        w0_value_b.shift(0.1*LEFT+0.05*UP)
        self.play(dial_1.animate.rotate(-0.1*60*DEGREES), Transform(w0_value, w0_value_b), run_time=1.0) #1.0 -> 0.9
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (0,0,0), 8.0)) #back to OG view
        self.wait()

        #Now how dafuq do I change w0 values nicely in the equation. 
        eq2b=Tex("=(-1)(-0.9) + (-1)(1) + 1")
        eq2b.match_style(eq2)
        eq2b.move_to(eq2)
        # self.play(TransformMatchingTex(eq2, eq2b), run_time=1.0)
        # self.wait()

        eq3b=Tex("=0.9")
        eq3b.match_style(eq3)
        eq3b.move_to(eq3)

        eq5b=Tex("=-1-0.9=-1.9")
        eq5b.match_style(eq5)
        eq5b.move_to(eq5)

        eq6b=Tex("Error^2=(y-\hat{y})^2=3.61")
        eq6b.match_style(eq6)
        eq6b.move_to(eq6)
        self.play(TransformMatchingTex(eq2, eq2b), TransformMatchingTex(eq3, eq3b), TransformMatchingTex(eq5, eq5b), TransformMatchingTex(eq6, eq6b), run_time=2.0)
        self.wait()

        ##Ok now plot was we make little moves
        axes2 = Axes(
            x_range=[-1.0, 2.1, 0.5],
            y_range=[0, 5, 1],
            axis_config={"include_ticks":True, "color": CHILL_BROWN, "stroke_width": 2,
                "include_tip":True, "tip_config": {"fill_opacity": 1, "width": 0.1,"length": 0.1}},
            x_axis_config={"include_numbers":True,  
                           "decimal_number_config":{"num_decimal_places":1, "font_size":30}},
            height=4, 
            width=6)
        axes2.add_coordinate_labels()

        axes2.move_to(RIGHT*3.2+0.1*DOWN)
        x_label_2=Tex('w_0', font_size=28).set_color(CHILL_BROWN)
        x_label_2.next_to(axes2, RIGHT)
        x_label_2.shift(2*DOWN+0.2*LEFT)

        y_label_2=Tex('Error^2', font_size=28).set_color(CHILL_BROWN)
        y_label_2.next_to(axes2, TOP)
        y_label_2.shift(0.85*DOWN-0.5*RIGHT) #+0.5*RIGHT)

        self.play(*[FadeOut(o) for o in [eq1,eq4, eq2b, eq3b, eq5b, eq6b]]+[FadeIn(axes2), FadeIn(x_label_2), FadeIn(y_label_2)])
        self.wait()

        #Now fade out first graph and move up dials
        dial_group=Group(dial_ticks_1, dial_1, w0_label, w0_value,dial_ticks_3, dial_3, b_label, b_value, dial_ticks_2, dial_2, w1_label, w1_value)
        og_graph_group=VGroup(axes, x_label, y_label, t1, t1l, t2, t2l, t3, t3l, t4, t4l, dots, dots_labels, label_box, label_t1, label_t2, label_t3)
        

        self.play(FadeOut(og_graph_group), dial_group.animate.shift(3*UP), run_time=3)
        self.wait()


        y=-1
        w0_value_c=None
        all_dots=VGroup()
        self.remove(w0_value_b); self.remove(w0_value)
        for i, w0 in enumerate(np.arange(-0.9, 2.0, 0.1)):
            yhat=-1*w0-1*w1+b
            E=(y-yhat)
            e=Dot(axes2.c2p(w0, E**2)).set_color(BLUE).scale(0.9)
            self.add(e)
            if i>0: dial_1.rotate(-0.1*60*DEGREES)
            all_dots.add(e)

            if w0_value_c is not None: self.remove(w0_value_c)
            w0_value_c=Tex(f'{w0:.1f}', color=WHITE)
            w0_value_c.scale(0.7).next_to(w0_label, RIGHT, aligned_edge=BOTTOM)
            w0_value_c.shift(0.1*LEFT+0.05*UP)
            self.add(w0_value_c)
            self.wait(0.1)

        self.wait()

        # Ok now 3d Paragraph 48 - let's go!
        # Alright, give what I know about manim here - I think I should actlly do this animation in 2 parts 
        # and put them together in premiere. I'll go ahead and do dials here, and do the 3d part of p48 in a new animation
        # This will give me indpendent camera control. 
        self.play(*[FadeOut(o) for o in [axes2, x_label_2, y_label_2, all_dots]])
        self.wait()

        w1_value_c=None
        #Set these to -2
        dial_1.rotate(4.1*60*DEGREES)
        dial_2.rotate(3.1*60*DEGREES)
        all_dots=VGroup()
        self.remove(w0_value_c); self.remove(w1_value); 
        for i, w0 in enumerate(np.arange(-1.9, 2.0, 0.25)):
            self.remove(w0_value_c)
            w0_value_c=Tex(f'{w0:.1f}', color=WHITE)
            w0_value_c.scale(0.7).next_to(w0_label, RIGHT, aligned_edge=BOTTOM)
            w0_value_c.shift(0.1*LEFT+0.05*UP)
            dial_1.rotate(-0.1*60*DEGREES)
            self.add(w0_value_c)

            for j, w1 in enumerate(np.arange(-1.9, 2.0, 0.25)):
                if w1_value_c is not None: self.remove(w1_value_c)
                w1_value_c=Tex(f'{w1:.1f}', color=WHITE)
                w1_value_c.scale(0.7).next_to(w1_label, RIGHT, aligned_edge=BOTTOM)
                w1_value_c.shift(0.1*LEFT+0.05*UP)
                dial_2.rotate(-0.1*60*DEGREES)
                self.add(w1_value_c)
                self.wait(1./30)

            dial_2.rotate(3.9*60*DEGREES)

        self.wait()
        self.wait(20)



class P48_50_3Dv3(InteractiveScene):
    def construct(self):

        axes3d = ThreeDAxes(
            x_range=[-2.0, 2.0, 0.5],
            y_range=[-2.0, 2.0, 0.5],  # Range for w1
            z_range=[0, 10, 2],
            height=6,
            width=6,
            depth=4,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        ).scale(0.8)


        # Add labels
        x_label = Tex('w_0', font_size=28).set_color(CHILL_BROWN)
        y_label = Tex('w_1', font_size=28).set_color(CHILL_BROWN)
        z_label = Tex('Error^2', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes3d.x_axis, RIGHT)
        y_label.next_to(axes3d.y_axis, UP)
        z_label.next_to(axes3d.z_axis, OUT)
        z_label.rotate(90*DEGREES, [1,0,0])

        self.add(axes3d) #, x_label, y_label, z_label)
        self.frame.reorient(-29, 59, 0, (-0.11, 0.15, 1.36), 8.00)
        # reorient(59, 68, 0, (0.38, 0.14, 0.4))

        # Create points
        dots = Group()
        b = 1   # Bias term
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        y = np.array([[-1], [1], [1], [1]])

        # Calculate total number of points for camera movement interpolation
        w0_range = np.arange(-1.9, 2.0, 0.25) #CRANK UP DENSITY FOR FINAL VIZ, AND MAKE SURE IT MATCHES DIAL TURNS ABOVE
        w1_range = np.arange(-1.9, 2.0, 0.25)
        total_points = len(w0_range) * len(w1_range)
        point_count = 0

        # Set initial and final camera orientations
        initial_orientation = (-29, 59, 0)
        final_orientation = (59, 68, 0)

        for w0 in w0_range:
            for w1 in w1_range:
                yhat = X[:,0]*w0 + X[:,1]*w1 + b
                error = np.mean((y.ravel()-yhat)**2)
                point = axes3d.c2p(w0, w1, error)
                dot = Sphere(radius=0.05, color='#3d5c6f', opacity=1.0).move_to(point)
                dots.add(dot)
                
                # # Calculate interpolated camera angles
                # progress = point_count / total_points
                # current_theta = initial_orientation[0] + (final_orientation[0] - initial_orientation[0]) * progress
                # current_phi = initial_orientation[1] + (final_orientation[1] - initial_orientation[1]) * progress
                # current_gamma = initial_orientation[2] + (final_orientation[2] - initial_orientation[2]) * progress
                
                # Update camera position
                # self.frame.reorient(current_theta, current_phi, current_gamma, (-0.11, 0.15, 1.36), 8.00) #This is looking dope!
                
                self.add(dot)
                # self.wait(1./30)
                point_count += 1

        self.wait()
        self.frame.reorient(47, 45, 0, (0.02, 0.18, 1.36), 8.00)
        self.wait()
        self.frame.reorient(52, 66, 0, (-0.04, 0.33, 1.73), 7.10)
        self.wait()
        self.frame.reorient(27, 47, 0, (0.0, -0.07, 1.54), 7.10)
        self.wait()
        self.frame.reorient(68, 61, 0, (-0.11, -0.21, 1.62), 7.09)
        self.wait()
        self.frame.reorient(90, 0, 0, (0.00, 0.00, 0.00), 8.6)
        self.wait()








        # ##Ok this is looking dope -> now I need to add incremental arrows downhill for the gradient. Claude might be able to help me. 
        # # After your existing code, add the gradient descent path
        # start_w0, start_w1 = -1.9, -1.9  # Starting point
        # learning_rate = 0.1  # Step size for gradient descent
        # num_steps = 8  # Number of arrows to show

        # arrows = VGroup()
        # current_w0, current_w1 = start_w0, start_w1

        # for i in range(num_steps):
        #     # Current position
        #     yhat_current = X[:,0]*current_w0 + X[:,1]*current_w1 + b
        #     error_current = np.mean((y.ravel()-yhat_current)**2)
            
        #     # Calculate gradients
        #     epsilon = 0.01
        #     # Gradient for w0
        #     yhat_w0_plus = X[:,0]*(current_w0 + epsilon) + X[:,1]*current_w1 + b
        #     yhat_w0_minus = X[:,0]*(current_w0 - epsilon) + X[:,1]*current_w1 + b
        #     grad_w0 = (np.mean((y.ravel()-yhat_w0_plus)**2) - np.mean((y.ravel()-yhat_w0_minus)**2)) / (2*epsilon)
            
        #     # Gradient for w1
        #     yhat_w1_plus = X[:,0]*current_w0 + X[:,1]*(current_w1 + epsilon) + b
        #     yhat_w1_minus = X[:,0]*current_w0 + X[:,1]*(current_w1 - epsilon) + b
        #     grad_w1 = (np.mean((y.ravel()-yhat_w1_plus)**2) - np.mean((y.ravel()-yhat_w1_minus)**2)) / (2*epsilon)
            
        #     # Update weights
        #     next_w0 = current_w0 - learning_rate * grad_w0
        #     next_w1 = current_w1 - learning_rate * grad_w1
            
        #     # Calculate next position's error
        #     yhat_next = X[:,0]*next_w0 + X[:,1]*next_w1 + b
        #     error_next = np.mean((y.ravel()-yhat_next)**2)
            
        #     # Create arrow
        #     start_point = axes3d.c2p(current_w0, current_w1, error_current)
        #     end_point = axes3d.c2p(next_w0, next_w1, error_next)
            
        #     arrow = Arrow(
        #         start_point,
        #         end_point,
        #         buff=0,
        #         thickness=3
        #     ) #.set_shade_in_3d(True)
        #     arrow.set_color(YELLOW)
        #     arrows.add(arrow)
            
        #     # Update current position
        #     current_w0, current_w1 = next_w0, next_w1

        # self.wait()
        # # self.frame.reorient(58, 67, 0, (-0.11, 0.15, 1.36))
        # # self.add(arrows[0])
        # self.play(self.frame.animate.reorient(54, 54, 0, (-0.31, -0.56, 2.16), 5.24), run_time=4)
        # self.wait()
        # self.add(arrows[0])
        # self.wait()
        # self.play(self.frame.animate.reorient(85, 62, 0, (-0.31, -0.3, 2.18), 5.89), run_time=4)

        # # Animate the arrows appearing one by one
        # for arrow in arrows[1:]:
        #     self.add(arrow)
        #     self.wait(0.2)
        # self.wait()

        # #I was thinkig I would move camera while adding arrows, but maybe this is fine. 
        # self.play(self.frame.animate.reorient(19, 44, 0, (0.38, 0.14, 0.4)), run_time=4.0)
        # self.wait()

        # # Ok, back at a kinda tricky 2d/3d hand off. 
        # # For paragraph 51, I want to basically go back to 2d -> I think I can do this by panning this to a 2d side view, 
        # # Then drawing in a nice parabolic curve
        # # Then I'll jump to a new 2d view, and maek sure the 2d parabolic curve is the same -> everythin else can change 
        # # with a crossfade in editing -> fade out 3d axes and fade in 2d axes. 
        # # For extra credit might be nice to fade out the axes here and just leave the curve that can hlep with cross disolved. 

        # self.play(self.frame.animate.reorient(0, 100, 0, (0.16, 0.1, 2.1), 8.00), FadeOut(arrows), run_time=4.0) #Pan to straight on view
        # #Now, draw in a nice parabola for one of the slices - claude can probably help!


        # # Create a parametric function for the error curve at w1=0
        # def error_at_w1_zero(w0):
        #     b = 1  # Bias term from your original code
        #     X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        #     y = np.array([[-1], [1], [1], [1]])
        #     w1 = 0  # Fixed slice
        #     yhat = X[:,0]*w0 + X[:,1]*w1 + b
        #     return np.mean((y.ravel()-yhat)**2)

        # # Create the curve
        # w0_points = np.linspace(-2.0, 2.0, 100)  # Matches your x-axis range
        # curve_points = []
        # for w0 in w0_points:
        #     point = axes3d.c2p(w0, 0, error_at_w1_zero(w0))
        #     curve_points.append(point)

        # slice_curve = VMobject()
        # slice_curve.set_points_smoothly(curve_points)
        # slice_curve.set_color(BLUE)
        # slice_curve.set_stroke(width=4)

        # # Animate the curve drawing
        # self.play(
        #     ShowCreation(slice_curve),
        #     FadeOut(dots),
        #     run_time=2.0
        # )
        # self.wait()

        # self.play(self.frame.animate.reorient(4, 98, 0, (-0.99, 0.14, 1.42), 4.40), run_time=2.0)
        # self.wait(20)

        ##Ok let's zoom in and then cut to illustrator



        # # Create points
        # dots = Group()
        # # y = -1  # Your target value*
        # b = 1   # Bias term
        # X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        # y = np.array([[-1], [1], [1], [1]])

        # for w0 in np.arange(-1.9, 2.0, 0.5):
        #     for w1 in np.arange(-1.9, 2.0, 0.5):
        #         yhat=X[:,0]*w0+X[:,1]*w1+b
        #         error=np.mean((y.ravel()-yhat)**2)
        #         # yhat = -1 * w0 - 1 * w1 + b
        #         # error = (y - yhat)**2
        #         point = axes3d.c2p(w0, w1, error)
        #         dot = Sphere(radius=0.05, color=BLUE, opacity=1).move_to(point)
        #         dots.add(dot)
        #         self.add(dot)
        #         self.wait(1./30)

        # self.wait()
        # self.play(self.frame.animate.reorient(38, 72, 0, (0.16, 0.56, 1.25)))



class InitialHackingP47(InteractiveScene):
    def construct(self):

        w0, w1, b = -1, 1, 1

        dial_ticks_1 = ImageMobject('/Users/stephen/manim_videos/_2025/perceptron/single_dial_ticks_brown.png')

        dial_ticks_1.scale(0.45)
        dial_ticks_1.move_to(DOWN*2.5+LEFT*5)
        w0_label=Tex('w_0', color=WHITE)
        w0_label.next_to(dial_ticks_1, 0.01*DOWN)

        dial_1=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_1.scale(0.32)
        dial_1.move_to(dial_ticks_1)
        dial_1.shift(0.075*DOWN+0.01*RIGHT)
        self.add(dial_ticks_1, dial_1, w0_label)


        dial_ticks_2=dial_ticks_1.copy()
        dial_ticks_2.shift(1.8*RIGHT)
        w1_label=Tex('w_1', color=WHITE)
        w1_label.next_to(dial_ticks_2, 0.01*DOWN)
        dial_2=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_2.scale(0.32)
        dial_2.move_to(dial_ticks_2)
        dial_2.shift(0.075*DOWN+0.01*RIGHT)
        self.add(dial_ticks_2, dial_2, w1_label)

        dial_ticks_3=dial_ticks_2.copy()
        dial_ticks_3.shift(1.8*RIGHT)
        b_label=Tex('b', color=WHITE)
        b_label.next_to(dial_ticks_3, 0.01*DOWN)
        dial_3=VGroup(Circle(stroke_color=WHITE), Line(ORIGIN, UP, stroke_color=WHITE))
        dial_3.scale(0.32)
        dial_3.move_to(dial_ticks_3)
        dial_3.shift(0.075*DOWN+0.01*RIGHT)
        self.add(dial_ticks_3, dial_3, b_label)




        # self.remove(dial_1)
        # Ok, not perfect bu workable, I can come back and add more dials
        # Let me work on rotation next
        # dial_1.rotate(45*DEGREES)
        # self.play(Rotate(dial_1, angle=45*DEGREES, about_point=dial_1.get_center()), run_time=2) #NICE

        ## Decision Boundary
        # Create the coordinate plane
        axes = Axes(
            x_range=[-1.5, 1.5, 1.0],
            y_range=[-1.5, 1.5, 1.0],
            axis_config={
                # "big_tick_numbers":[-1,1],
                "include_ticks":False,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip":True,
                "tip_config": {  # Changed from tip_shape
                    "fill_opacity": 1,
                    "width": 0.1,
                    "length": 0.1
                }
            },
            x_axis_config={"include_numbers":True, 
               #"big_tick_spacing":0.5
               "decimal_number_config":{"num_decimal_places":1, "font_size":30}},
            # x_axis_config={"numbers_with_elongated_ticks": [-1, 1]}
        )
        axes.scale(1.5)
        axes.move_to(LEFT*3+UP)  # Moves it 2 units to the right
        self.add(axes)



        # Ok hacked on putting ticks at +/-1 for a while, couldn't get it, also can't figure out how to put arrows
        # on both ends of the axes - both of these are mildly annoying, but not that big of a deal, skipping for now. 

        

        x_at_ymin = -(-1.3*w1 + b)/w0  # when y = -1.3
        x_at_ymax = -(1.3*w1 + b)/w0   # when y = 1.3
        x_min = max(-1.3, min(x_at_ymin, x_at_ymax))
        x_max = min(1.3, max(x_at_ymin, x_at_ymax))

        line = axes.get_graph(
            partial(get_decision_boundary_value, w0=w0, w1=w1, b=b),
            x_range=[x_min, x_max], color=WHITE)
        self.add(line)

        # Create the regions
        above_region = VMobject()
        above_region.set_points_as_corners(get_region_points(axes, w0=w0, w1=w1, b=b, above=True))
        above_region.set_fill(YELLOW, opacity=0.35)
        above_region.set_stroke(width=0)

        below_region = VMobject()
        below_region.set_points_as_corners(get_region_points(axes,w0=w0, w1=w1, b=b,  above=False))
        below_region.set_fill(BLUE, opacity=0.35)
        below_region.set_stroke(width=0)

        # Add the regions (add these before the line so the line appears on top)
        self.add(above_region, below_region)
        dots = VGroup()
        dots.add(Dot(axes.c2p(-1, -1), radius=0.05).set_color(BLUE))
        dots.add(Dot(axes.c2p(-1, 1), radius=0.05).set_color(YELLOW))
        dots.add(Dot(axes.c2p(1, -1), radius=0.05).set_color(YELLOW))
        dots.add(Dot(axes.c2p(1, 1), radius=0.05).set_color(YELLOW))
        self.add(dots)

        
        ## 1D Loss as a funciton of w0
        axes2 = Axes(
            x_range=[-1.5, 1.6, 0.5],
            y_range=[0, 5, 1],
            axis_config={
                # "big_tick_numbers":[-1,1],
                "include_ticks":True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip":True,
                "tip_config": {  # Changed from tip_shape
                    "fill_opacity": 1,
                    "width": 0.1,
                    "length": 0.1
                }
            },
            x_axis_config={"include_numbers":True, 
                           #"big_tick_spacing":0.5
                           "decimal_number_config":{"num_decimal_places":1, "font_size":30}},
            # # y_axis_config={"include_numbers":True, "line_to_number_direction": UP},
            # y_axis_config={
            #     "include_numbers": True,
            #     "decimal_number_config": {
            #         "num_decimal_places": 0,
            #         "font_size": 30,
            #         #"label_direction": 0  # This will make numbers vertical
            #     },
            #     #"excluding_origin": True  # This removes the zero label
            #     "numbers_to_exclude": [0],
            #     # "label_direction": 0 
            # },
            height=4, 
            width=6
            # x_axis_config={"numbers_with_elongated_ticks": [-1, 1]}
        )

        axes2.add_coordinate_labels()

        axes2.move_to(RIGHT*3.2+0.1*DOWN)
        x_label=Tex('w_0', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes2, RIGHT)
        x_label.shift(2*DOWN+0.2*LEFT)

        y_label=Tex('Error^2', font_size=28).set_color(CHILL_BROWN)
        y_label.next_to(axes2, TOP)
        y_label.shift(0.85*DOWN+0.5*RIGHT)


        #Compute error for -1, -1 point
        yhat=-1*w0-1*w1+b
        y=-1
        E=(y-yhat)
        print(E)

        #Plot point
        e1=Dot(axes2.c2p(w0, E**2)).set_color(BLUE).scale(0.9)




        self.add(axes2, x_label, y_label, e1)



        # self.remove(axes2)

        self.embed()











