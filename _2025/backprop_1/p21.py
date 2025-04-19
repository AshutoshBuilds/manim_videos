from manimlib import *
from functools import partial
import sys
sys.path.append('/Users/stephen/manim_videos/welch_assets')
from welch_axes import *

sys.path.append('/Users/stephen/manim_videos/_2025/backprop_1')
from backprop_data import xs1, losses1

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

def get_x_axis(x_max):
    x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(0, x_max, min_ticks=3, max_ticks=16)
    x_axis=WelchXAxis(        
        x_ticks=x_ticks,  
        x_tick_height=0.15,        
        x_label_font_size=24,           
        stroke_width=3, 
        arrow_tip_scale=0.1,
        x_min=0,
        x_max=x_max
    )
    return x_axis

class P21(InteractiveScene):
    '''
    Ok I'm thinking that the version of this scene where we zoom out on an alrady constructed big Graph might be easier
    than growing the graph in place. The weird thing will be x-axis spacing etc, that will take some noodling. 
    Hmm figuring out how to handle the axes while I do the big zoom out is tricker than expected
    I think it's probably worth figuring out, the laternative ot to do maptlotlib, but I really want the 
    buttery manim animation start/ends, and I think these are components I will reuse. 

    '''
    def construct(self):

        initial_length=3
        l_tracker = ValueTracker(initial_length)

        x_axis = always_redraw(lambda: get_x_axis(l_tracker.get_value()))

        self.add(x_axis)
        self.wait()

        self.play(l_tracker.animate.set_value(20), run_time=2)
        self.wait()




        x_axis=WelchXAxis(        
            x_ticks=[-0.003, 0.007, 0.017],  
            x_tick_height=0.15,        
            x_label_font_size=24,           
            stroke_width=3, 
            arrow_tip_scale=0.1
        )

        y_axis=WelchYAxis(  
            y_ticks=[0.3901, 0.3916, 0.3930],  
            y_tick_width=0.15,        
            y_label_font_size=24,            
            stroke_width=3,          
            arrow_tip_scale=0.1,      
        )



        x_ticks, x_axis_min, x_axis_max=generate_nice_ticks(0, 4, min_ticks=3, max_ticks=16)
        y_ticks, y_axis_min, y_axis_max=generate_nice_ticks(0, 2.0, min_ticks=3, max_ticks=16)

        x_axis=WelchXAxis(        
            x_ticks=x_ticks,  
            x_tick_height=0.15,        
            x_label_font_size=24,           
            stroke_width=3, 
            arrow_tip_scale=0.1,
            x_min=x_axis_min, 
            x_max=x_axis_max
        )

        y_axis=WelchYAxis(        
            y_ticks=y_ticks,  
            y_tick_width=0.15,        
            y_label_font_size=24,           
            stroke_width=3, 
            arrow_tip_scale=0.1,
            y_min=0, 
            y_max=y_axis_max
        )

        # Is it like i try to figure out how to smoothly animate the length of the axis and I have the tick marks automatically "fill in"?
        # Ok i tried that with claude at least, looks terrible lol. 
        # Hmm...
        # Ok I think it's worth looking at stephen_playground_og.py -> looks like grant showed me how to solve an analagous problem there. 
        

        # # x_axis.move_to([-2, -2, 0])
        initial_length=5
        l_tracker = ValueTracker(initial_length)

        self.add(x_axis) #, y_axis)

        self.play.animate(x_axis.animate.set_max_val(5), run_time=5)

        x_axis.set_max_val(5)
        # self.wait()

        x_axis.set_max_val(6)

        x_axis.set_max_val(7)


        x_axis.update_from_range(-1, 7)
        self.play.animate()
        # self.embed()



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
