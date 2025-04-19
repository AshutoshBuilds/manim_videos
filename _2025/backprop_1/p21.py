from manimlib import *
from functools import partial


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

class P21(InteractiveScene):
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

        self.add(axes)