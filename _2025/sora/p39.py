from manimlib import *
import numpy as np
from PIL import Image

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

class P39(Scene):
    def construct(self):
        cat = ImageMobject('n02123045_1955.jpg').scale(1)
        self.play(FadeIn(cat))
        self.wait(1)
        
        img = Image.open('cat_downscaled.jpg')
        img_array = np.array(img)
        num_pixels = (32, 32) 
        
        img_resized = img.resize(num_pixels, Image.NEAREST)
        img_array = np.array(img_resized)
        pixel_dimensions = img_array.shape  
        height, width = img_array.shape[:2]
        
        pixel_squares = VGroup()
        
        pixel_size = cat.height / height
        
        for i in range(height):
            for j in range(width):
                if len(img_array.shape) == 3: 
                    r, g, b = img_array[i, j][:3]
                    color = rgb_to_color([r/255, g/255, b/255])
                else: 
                    gray = img_array[i, j]
                    color = rgb_to_color([gray/255, gray/255, gray/255])
                
                pixel_square = Square(side_length=pixel_size, fill_color=color, fill_opacity=1.0, stroke_width=0)
                
                x_pos = (j - width/2 + 0.5) * pixel_size
                y_pos = -(i - height/2 + 0.5) * pixel_size
                
                pixel_square.move_to([x_pos, y_pos, 0])
                pixel_squares.add(pixel_square)
        
        pixel_squares.move_to(cat.get_center())
        
        
        
        self.play(FadeOut(cat))
        
        
        self.play(FadeIn(pixel_squares))
        
        
        animations = []
        center = pixel_squares.get_center()
        gap_factor = 0.4  
        
        for pixel in pixel_squares:
            pixel_pos = pixel.get_center()
            direction_vector = pixel_pos - center
            
            distance = np.linalg.norm(direction_vector)
            
            if distance > 0:
                unit_vector = direction_vector / distance
                
                displacement = unit_vector * distance * gap_factor
                new_position = pixel_pos + displacement
                
                animations.append(ApplyMethod(pixel.move_to, new_position))
                
        new_pixel = VGroup()
        for pixel in pixel_squares:
            new_pixel.add(pixel)
            
                
        
        

        self.play(*animations, run_time=2)
        
        down = VGroup(pixel_squares[992], pixel_squares[1023])
        right = VGroup(pixel_squares[0], pixel_squares[1023])
        
        down_bracket = Brace(down, direction=DOWN)
        down_bracket_text = Tex("32 px").next_to(down_bracket, DOWN)
        right_bracket = Brace(right, direction=RIGHT)
        right_bracket_text = Tex("32 px").next_to(right_bracket, RIGHT)
        
        self.add(down_bracket, down_bracket_text, right_bracket, right_bracket_text)
        
        self.wait(2)
        
        self.remove(down_bracket, down_bracket_text, right_bracket, right_bracket_text)
        
        self.wait()
        
        self.play(pixel_squares.animate.shift(LEFT * 3.5))
        
        axes = SVGMobject('p39_2.svg')[1:]
        axes.move_to(ORIGIN).scale(4).shift(RIGHT * 3)
        
        
        x1 = VGroup(axes[6], axes[7])
        x2 = VGroup(axes[8], axes[9])
        x3 = VGroup(axes[10], axes[11])
        xn = VGroup(axes[15], axes[16])
        
        x1_tick = axes[12]
        x2_tick = axes[13]
        x3_tick = axes[14]
        
        dots = VGroup(axes[19], axes[20], axes[21])
        
        x1_arrow = VGroup(axes[2], axes[3])   
        x2_arrow = VGroup(axes[0], axes[1])
        x3_arrow = VGroup(axes[4], axes[5])
        xn_arrow = VGroup(axes[17], axes[18])
        
        final_point = axes[22]
        
        self.wait()
        
        self.play(FadeIn(x1_arrow), FadeIn(x2_arrow), FadeIn(x3_arrow), FadeIn(x1), FadeIn(x2), FadeIn(x3))
        
        self.wait()
        
        pixel_squares[0].set_stroke(WHITE, width=4)
        
        self.play(pixel_squares[0].animate.scale(3).next_to(x1_tick, DOWN), run_time = 1)
        self.play(
            LaggedStart(
            Indicate(x1_arrow),
            ReplacementTransform(pixel_squares[0], x1_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        pixel_squares[1].set_stroke(WHITE, width=4)
        
        self.play(pixel_squares[1].animate.scale(3).next_to(x2_tick, LEFT), run_time = 1)
        self.play(
            LaggedStart(
            Indicate(x2_arrow),
            ReplacementTransform(pixel_squares[1], x2_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        pixel_squares[2].set_stroke(WHITE, width=4)
        self.play(pixel_squares[2].animate.scale(3).next_to(x3_tick, DOWN), run_time = 1)
        self.play(
            LaggedStart(
            Indicate(x3_arrow),
            ReplacementTransform(pixel_squares[2], x3_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        other_pixels = VGroup(*[pixel_squares[i] for i in range(3, len(pixel_squares))])
        xn_group = VGroup(xn_arrow, xn, dots)
        self.play(
            LaggedStart(ReplacementTransform(other_pixels, final_point), FadeIn(xn_group), lag_ratio=0.75)
            , run_time = 3
        )
        
        self.play(axes.animate.move_to(ORIGIN))


        

        



        self.embed()
        
def label_with_indices(self, mobject, scale=0.4, color=YELLOW):
    for i, submob in enumerate(mobject.submobjects):
        label = Text(str(i)).scale(scale).set_color(color)
        label.move_to(submob.get_center())
        self.add(label)