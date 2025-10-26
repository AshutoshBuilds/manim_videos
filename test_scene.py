from manim_imports_ext import *

class TestScene(Scene):
    def construct(self):
        # Create a simple circle
        circle = Circle(radius=1, color=BLUE)
        circle.set_fill(BLUE, opacity=0.5)

        # Create some text
        text = Text("Hello, Manim!", font_size=48)
        text.set_color(YELLOW)

        # Position the text below the circle
        text.next_to(circle, DOWN, buff=1)

        # Add animations
        self.play(ShowCreation(circle), Write(text))
        self.wait(2)

        # Transform the circle
        square = Square(side_length=2, color=RED)
        square.set_fill(RED, opacity=0.5)

        self.play(Transform(circle, square))
        self.wait(2)
