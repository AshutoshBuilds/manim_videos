from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class Line3D(VMobject):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners([start, end])

class Dot3D(Sphere):
    def __init__(self, point=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(point)


class P24v1(InteractiveScene):
    def construct(self):

        self.wait(20)
        self.embed()