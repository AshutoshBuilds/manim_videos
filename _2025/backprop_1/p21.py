from manimlib import *
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
WELCH_ASSET_PATH='/Users/stephen/manim_videos/welch_assets'

class WelchXAxis(VGroup):
    def __init__(
        self,
        x_ticks=[1, 2, 3, 4, 5],  # Default tick values
        x_tick_height=0.2,        # Default tick height
        x_label_font_size=24,     # Default font size
        tip_size=0.2,             # Default tip size
        stroke_width=2,           # Default stroke width
        color=CHILL_BROWN,              # Default color (using predefined BROWN)
        arrow_tip_scale=0.1, 
        **kwargs
    ):
        VGroup.__init__(self, **kwargs)
        
        # Store parameters
        self.x_ticks = x_ticks
        self.x_tick_height = x_tick_height
        self.x_label_font_size = x_label_font_size
        self.tip_size = tip_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale=arrow_tip_scale
        
        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        # Calculate the axis length based on the min and max tick values with padding
        x_min = min(self.x_ticks) - 0.5
        x_max = max(self.x_ticks) + 0.5
        
        # Create a line for the x-axis
        axis_line = Line(
            start=np.array([x_min, 0, 0]),
            end=np.array([x_max, 0, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        
        # Add arrow tip at the end using Arrow instead of add_tip
        #SW - HEY MAYBE WE ACTUALLY JUST GO AHEAD AND IMPORT AN ILLUSTRATOR SVG FOR THE TIP?
        arrow_tip=SVGMobject(WELCH_ASSET_PATH+'/welch_arrow_tip_1.svg')
        arrow_tip.scale(self.arrow_tip_scale)
        arrow_tip.move_to([x_max, 0, 0])
        
        self.axis_line = VGroup(axis_line, arrow_tip)
        self.add(self.axis_line)
        
    def _create_ticks(self):
        self.ticks = VGroup()
        
        for x_val in self.x_ticks:
            tick = Line(
                start=np.array([x_val, 0, 0]),
                end=np.array([x_val, -self.x_tick_height, 0]),  # Ticks extend downward
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = VGroup()
        
        for x_val in self.x_ticks:
            # In 3B1B's manim, use TexMobject instead of MathTex
            label = Tex(str(x_val))
            label.scale(self.x_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([x_val, -self.x_tick_height, 0]),
                DOWN,
                buff=0.1
            )
            self.labels.add(label)
            
        self.add(self.labels)
    
    # Helper method to get the axis line
    def get_axis_line(self):
        return self.axis_line
    
    # Helper method to get ticks
    def get_ticks(self):
        return self.ticks
    
    # Helper method to get labels
    def get_labels(self):
        return self.labels


class P21(InteractiveScene):
    def construct(self):

        x_axis=WelchXAxis(        
            x_ticks=[1, 2, 3, 4, 5],  
            x_tick_height=0.15,        
            x_label_font_size=24,   
            tip_size=0.2,          
            stroke_width=3
        )


        self.frame.add(x_axis)
        self.wait()

        self.embed()

        # axes = Axes(
        #     x_range=[-1.2, 1.2, 1.0],
        #     y_range=[-1.2, 1.2, 1.0],
        #     axis_config={
        #         # "big_tick_numbers":[-1,1],
        #         "include_ticks":True,
        #         "color": CHILL_BROWN,
        #         "stroke_width": 2,
        #         "include_tip":True,
        #         "include_numbers":True,
        #         "tip_config": {  # Changed from tip_shape
        #             "fill_opacity": 1,
        #             "width": 0.1,
        #             "length": 0.1
        #         }
        #     },
        #     # x_axis_config={"include_numbers":True, 
        #     #                #"big_tick_spacing":0.5
        #     #                "decimal_number_config":{"num_decimal_places":1, "font_size":30}},

        #     # x_axis_config={"numbers_with_elongated_ticks": [-1, 1]}
        # )
