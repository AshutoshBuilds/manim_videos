from manimlib import *
from functools import partial
import sys
sys.path.append('/Users/stephen/manim_videos/welch_assets')
from welch_axes import *

sys.path.append('/Users/stephen/manim_videos/_2025/backprop_1')
from backprop_data import xs1, losses1, all_probs_1

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

def get_x_axis(t, intial_bounds, final_bounds):
    lower_bound, upper_bound = time_to_bounds(t, intial_bounds, final_bounds)
    x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(lower_bound, 0.95*upper_bound, min_ticks=3, max_ticks=16, ignore=[])
    x_axis=WelchXAxis(
        x_min=lower_bound,
        x_max=upper_bound,      
        x_ticks=x_ticks,  
        x_tick_height=0.15,        
        x_label_font_size=24,           
        stroke_width=3, 
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5
    )
    return x_axis

def time_to_bounds(t, intial_bounds, final_bounds):
    lower_bound=t*(final_bounds[0]-intial_bounds[0])+intial_bounds[0]
    upper_bound=t*(final_bounds[1]-intial_bounds[1])+intial_bounds[1]
    return lower_bound, upper_bound

class P21(InteractiveScene):
    def construct(self):

        initial_x_range=[-0.027, 0.013]
        final_x_range=[-1.1, 4.1]

        indices_in_range=np.logical_and(xs1>initial_x_range[0], xs1<initial_x_range[1])

        initial_time=0.0
        t_tracker = ValueTracker(initial_time)

        x_axis = always_redraw(lambda: get_x_axis(t_tracker.get_value(), initial_x_range, final_x_range))

        self.add(x_axis)
        self.wait()

        self.play(t_tracker.animate.set_value(1.0), run_time=8)
        self.wait()









# class P21Hacking2(InteractiveScene):
#     '''
#     Ok I'm thinking that the version of this scene where we zoom out on an alrady constructed big Graph might be easier
#     than growing the graph in place. The weird thing will be x-axis spacing etc, that will take some noodling. 
#     Hmm figuring out how to handle the axes while I do the big zoom out is tricker than expected
#     I think it's probably worth figuring out, the laternative ot to do maptlotlib, but I really want the 
#     buttery manim animation start/ends, and I think these are components I will reuse. 

#     '''
#     def construct(self):

#         initial_time=3
#         t_tracker = ValueTracker(initial_time)

#         x_axis = always_redraw(lambda: get_x_axis(t_tracker.get_value()))

#         self.add(x_axis)
#         self.wait()

#         self.play(t_tracker.animate.set_value(50), run_time=2)
#         self.wait()



        # Ok this patter is pretty clunky, but I think it can work. 
        # I was thinking I could zoom out as I grew the axis, but 
        # That makes my ticks and numbers shrink!
        # I need to modify my axis to have like an in place mode, or just act that way be default
        # where it gets longer as far as teh tick marks are concerned, but stays the length 
        # on the canvas. basically I need to change the start and stop values/scale on the
        # a static line - that should be do-able. man this is more complicated than I thought. 



        # x_axis=WelchXAxis(        
        #     x_ticks=[-0.003, 0.007, 0.017],  
        #     x_tick_height=0.15,        
        #     x_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1
        # )

        # y_axis=WelchYAxis(  
        #     y_ticks=[0.3901, 0.3916, 0.3930],  
        #     y_tick_width=0.15,        
        #     y_label_font_size=24,            
        #     stroke_width=3,          
        #     arrow_tip_scale=0.1,      
        # )



        # x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(0, 4, min_ticks=3, max_ticks=16)
        # y_ticks, y_axis_min, y_axis_max=generate_nice_ticks(0, 2.0, min_ticks=3, max_ticks=16)

        # x_axis=WelchXAxis(        
        #     x_ticks=x_ticks,  
        #     x_tick_height=0.15,        
        #     x_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1,
        #     x_min=x_axis_min, 
        #     x_max=x_axis_max
        # )

        # y_axis=WelchYAxis(        
        #     y_ticks=y_ticks,  
        #     y_tick_width=0.15,        
        #     y_label_font_size=24,           
        #     stroke_width=3, 
        #     arrow_tip_scale=0.1,
        #     y_min=0, 
        #     y_max=y_axis_max
        # )

        # Is it like i try to figure out how to smoothly animate the length of the axis and I have the tick marks automatically "fill in"?
        # Ok i tried that with claude at least, looks terrible lol. 
        # Hmm...
        # Ok I think it's worth looking at stephen_playground_og.py -> looks like grant showed me how to solve an analagous problem there. 
        

        # # x_axis.move_to([-2, -2, 0])
        # initial_length=5
        # l_tracker = ValueTracker(initial_length)

        # self.add(x_axis) #, y_axis)

        # self.play.animate(x_axis.animate.set_max_val(5), run_time=5)

        # x_axis.set_max_val(5)
        # # self.wait()

        # x_axis.set_max_val(6)

        # x_axis.set_max_val(7)


        # x_axis.update_from_range(-1, 7)
        # self.play.animate()
        # # self.embed()



class AxisHacking(InteractiveScene):
    def construct(self):

        x_axis=WelchXAxis(        
            x_ticks=[1, 2, 3, 4, 5],  
            x_tick_height=0.15,        
            x_label_font_size=24,           
            stroke_width=3, 
            arrow_tip_scale=0.1
        )

        y_axis=WelchYAxis(  
            y_ticks=[1, 2, 3, 4, 5],  
            y_tick_width=0.15,        
            y_label_font_size=24,            
            stroke_width=3,          
            arrow_tip_scale=0.1,      
        )

        # # x_axis.move_to([-2, -2, 0])


        self.add(x_axis, y_axis)
        # self.wait()

        # self.embed()
