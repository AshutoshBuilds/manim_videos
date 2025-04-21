from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

class P24v1(InteractiveScene):
    def construct(self):

    	# Create the surface
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 2.0, 0.5],
            height=8,
            width=8,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )



        # Add labels
        x_label = Tex(r'\theta_{1}', font_size=28).set_color(CHILL_BROWN)
        y_label = Tex(r'\theta_{2}', font_size=28).set_color(CHILL_BROWN)
        z_label = Tex('Loss', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, RIGHT)
        y_label.next_to(axes.y_axis, UP)
        z_label.next_to(axes.z_axis, OUT)
        z_label.rotate(90*DEGREES, [1,0,0])


        self.add(axes, x_label, y_label, z_label)