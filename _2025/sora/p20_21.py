from manimlib import *
from PIL import Image
import numpy as np
import math
 
COOL_GREEN = '#6c946f'
CHILL_BROWN='#948979'
COOL_YELLOW = '#ffd35a'

'''
import torch
stephen_hat = Image.open('me_with_hat.jpeg')
stephen_no_hat = Image.open('me_no_hat_cropped_1.jpeg')

device = torch.device("cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.no_grad():
    stephen_no_hat_inputs = processor(images=stephen_no_hat, return_tensors="pt").to(device)
    stephen_hat_inputs = processor(images=stephen_hat, return_tensors="pt").to(device)

    image_features_hat = model.get_image_features(**stephen_no_hat_inputs)
    image_features_no_hat = model.get_image_features(**stephen_hat_inputs)

    image_features_hat = F.normalize(image_features_hat, dim=-1)
    image_features_no_hat = F.normalize(image_features_no_hat, dim=-1)
    
def compute_similarity(text):
    with torch.no_grad():
        text_inputs = processor(text=[text], return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        similarity_stephen_hat = (image_features_hat @ text_features.T).squeeze().item()
        similarity_stephen_no_hat = (image_features_no_hat @ text_features.T).squeeze().item()  
        
        print(f"Text: '{text}'")
        print(f"Cosine similarity with 'me_with_hat.jpeg': {similarity_stephen_hat:.4f}")
        print(f"Cosine similarity with 'me_no_hat_cropped_1.jpeg': {similarity_stephen_no_hat:.4f}")
'''

def line_intersection(p1, p2, p3, p4):
    A = np.array([
        [p2[0] - p1[0], p3[0] - p4[0]],
        [p2[1] - p1[1], p3[1] - p4[1]]
    ])
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    
    try:
        t, s = np.linalg.solve(A, b)
        intersection = p1 + t * (p2 - p1)
        return intersection
    except np.linalg.LinAlgError:
        return None

class P20(InteractiveScene):
    def construct(self):
        stephen_no_hat = ImageMobject("me_no_hat_cropped_1.jpeg").scale(0.55)         
        stephen_hat = ImageMobject("me_with_hat.jpeg").scale(0.55)

        Group(stephen_no_hat, stephen_hat).arrange(DOWN, buff=1).shift(LEFT * 5.5)
        
        arrows = SVGMobject("p_20_21_to_manim-03.svg")[1:].scale(6)
        top_left_arrow = arrows[0].next_to(stephen_no_hat, RIGHT)
        bottom_left_arrow = arrows[1].next_to(stephen_hat, RIGHT)
        
        top_image_encoder_text = SVGMobject("top_image_encoder.svg")[3:].scale(5).next_to(top_left_arrow, RIGHT)
        bottom_image_encoder_text = SVGMobject("bottom_image_encoder.svg")[3:].scale(5).next_to(bottom_left_arrow, RIGHT)
        
        
        
        scale_factor = 0.002
        points = [
            scale_factor * np.array([493.37, 196.07, 0]),
            scale_factor * np.array([493.37, 458.95, 0]),
            scale_factor * np.array([670.56, 415.78, 0]),
            scale_factor * np.array([670.56, 240.21, 0]),
        ]

        rhombus = Polygon(*points,
                          fill_color="#6c946f",
                          fill_opacity=0.2,
                          stroke_color="#6c946f",
                          stroke_opacity=1,
                          stroke_width=2).scale(5)
        
        top_image_encoder_outer = rhombus.copy().move_to(top_image_encoder_text.get_center())

        bottom_image_encoder_outer = rhombus.copy().move_to(bottom_image_encoder_text.get_center())
        
        top_image_encoder = VGroup(top_image_encoder_outer, top_image_encoder_text)

        bottom_image_encoder = VGroup(bottom_image_encoder_outer, bottom_image_encoder_text)
        
        top_right_arrow = arrows[2].next_to(top_image_encoder, RIGHT)
        bottom_right_arrow = arrows[3].next_to(bottom_image_encoder, RIGHT)
        
        stephen_no_hat_equation = Tex(r"I_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(COOL_GREEN).next_to(top_right_arrow, RIGHT).scale(0.9).shift(LEFT * 0.3)

        stephen_hat_equation = Tex(r"\hat{I}_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(COOL_GREEN).next_to(bottom_right_arrow, RIGHT).scale(0.9).shift(LEFT * 0.3)

        self.play(FadeIn(stephen_hat), FadeIn(stephen_no_hat))
        self.play(GrowFromEdge(top_left_arrow, LEFT), GrowFromEdge(bottom_left_arrow, LEFT))
        self.play(ShowCreation(top_image_encoder), ShowCreation(bottom_image_encoder))
        self.play(GrowFromEdge(top_right_arrow, LEFT), GrowFromEdge(bottom_right_arrow, LEFT))
        self.play(Write(stephen_no_hat_equation), Write(stephen_hat_equation))
        self.embed()

class P21(InteractiveScene):
    def construct(self):
        x_axis = WelchXAxis(0, 8).move_to(ORIGIN)
        y_axis = WelchYAxis(0, 8).move_to(ORIGIN)
        x_axis.ticks.set_opacity(0)
        x_axis.labels.set_opacity(0)
        y_axis.ticks.set_opacity(0)
        y_axis.labels.set_opacity(0)
        
        axes = VGroup(x_axis, y_axis)
        axes.shift(LEFT * 4.5)
        
        stephen_no_hat = ImageMobject("me_no_hat_cropped_1.jpeg").scale(0.3)         
        stephen_hat = ImageMobject("me_with_hat.jpeg").scale(0.3)
        
        stephen_hat.move_to((y_axis.get_center() + y_axis.get_top())/2).shift(LEFT * 0.6)
        stephen_no_hat.move_to((x_axis.get_center() + x_axis.get_right())/2).shift(DOWN * 0.6)
        
        # Define the angle (in radians) to rotate the arrow up (e.g., 15 degrees)

        # Create the arrows
        stephen_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=6).set_color(COOL_GREEN)
        stephen_no_hat_arrow = WelchXAxis(x_min=0, x_max=2.25, axis_length_on_canvas=2.25, stroke_width=6).set_color(COOL_GREEN)
        
        
        
        
        stephen_hat_arrow.ticks.set_opacity(0)
        stephen_hat_arrow.labels.set_opacity(0)
        stephen_no_hat_arrow.ticks.set_opacity(0)
        stephen_no_hat_arrow.labels.set_opacity(0)
        
        axes_intersection = line_intersection(
            x_axis.get_axis_line().get_left(), x_axis.get_axis_line().get_right(),
            y_axis.get_axis_line().get_bottom(), y_axis.get_axis_line().get_top()
        )

        stephen_hat_arrow.shift(axes_intersection - stephen_hat_arrow.get_axis_line().get_left())
        stephen_no_hat_arrow.shift(axes_intersection - stephen_no_hat_arrow.get_axis_line().get_left())
        
        stephen_hat_arrow.rotate(65 * DEGREES, about_point=axes_intersection)
        stephen_no_hat_arrow.rotate(15 * DEGREES, about_point=axes_intersection)
        
        stephen_delta_arrow_length = math.sqrt(
            (stephen_hat_arrow.arrow.get_top()[0] - stephen_no_hat_arrow.arrow.get_right()[0]) ** 2 +
            (stephen_hat_arrow.arrow.get_top()[1] - stephen_no_hat_arrow.arrow.get_right()[1]) ** 2
                                               )
        stephen_delta_arrow = WelchXAxis(
            x_min=0, x_max=1, axis_length_on_canvas=stephen_delta_arrow_length, stroke_width=6
        ).set_color(COOL_YELLOW)

        stephen_delta_arrow.ticks.set_opacity(0)
        stephen_delta_arrow.labels.set_opacity(0)
        
        stephen_hat_arrow_label = Tex(r"I_{man}").set_color(COOL_GREEN).next_to(stephen_hat_arrow.arrow, UP)
        stephen_no_hat_arrow_label = Tex(r"\hat{I}_{man}").set_color(COOL_GREEN).next_to(stephen_no_hat_arrow.arrow, RIGHT)
        
        self.add(axes)
        self.add(stephen_hat, stephen_no_hat)
        self.add(stephen_hat_arrow, stephen_no_hat_arrow)
        self.add(stephen_hat_arrow_label, stephen_no_hat_arrow_label)
        self.embed()
    
class P20_P21(InteractiveScene):
    def construct(self):
        stephen_no_hat = ImageMobject("me_no_hat_cropped_1.jpeg").scale(0.55)         
        stephen_hat = ImageMobject("me_with_hat.jpeg").scale(0.55)

        Group(stephen_no_hat, stephen_hat).arrange(DOWN, buff=1).shift(LEFT * 5.5)
        
        arrows = SVGMobject("p_20_21_to_manim-03.svg")[1:].scale(6)
        top_left_arrow = arrows[0].next_to(stephen_no_hat, RIGHT)
        bottom_left_arrow = arrows[1].next_to(stephen_hat, RIGHT)
        
        top_image_encoder_text = SVGMobject("top_image_encoder.svg")[3:].scale(5).next_to(top_left_arrow, RIGHT)
        bottom_image_encoder_text = SVGMobject("bottom_image_encoder.svg")[3:].scale(5).next_to(bottom_left_arrow, RIGHT)
        
        
        
        scale_factor = 0.002
        points = [
            scale_factor * np.array([493.37, 196.07, 0]),
            scale_factor * np.array([493.37, 458.95, 0]),
            scale_factor * np.array([670.56, 415.78, 0]),
            scale_factor * np.array([670.56, 240.21, 0]),
        ]

        rhombus = Polygon(*points,
                          fill_color="#6c946f",
                          fill_opacity=0.2,
                          stroke_color="#6c946f",
                          stroke_opacity=1,
                          stroke_width=2).scale(5)
        
        top_image_encoder_outer = rhombus.copy().move_to(top_image_encoder_text.get_center())

        bottom_image_encoder_outer = rhombus.copy().move_to(bottom_image_encoder_text.get_center())
        
        top_image_encoder = VGroup(top_image_encoder_outer, top_image_encoder_text)

        bottom_image_encoder = VGroup(bottom_image_encoder_outer, bottom_image_encoder_text)
        
        top_right_arrow = arrows[2].next_to(top_image_encoder, RIGHT)
        bottom_right_arrow = arrows[3].next_to(bottom_image_encoder, RIGHT)
        
        stephen_no_hat_equation = Tex(r"I_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(COOL_GREEN).next_to(top_right_arrow, RIGHT).scale(0.9).shift(LEFT * 0.3)

        stephen_hat_equation = Tex(r"\hat{I}_{man} = [-0.13\ -0.10\ \cdots\ -0.56]").set_color(COOL_GREEN).next_to(bottom_right_arrow, RIGHT).scale(0.9).shift(LEFT * 0.3)

        self.play(FadeIn(stephen_hat), FadeIn(stephen_no_hat))
        self.play(DrawBorderThenFill(top_left_arrow), DrawBorderThenFill(bottom_left_arrow))
        self.play(ShowCreation(top_image_encoder), ShowCreation(bottom_image_encoder))
        self.play(DrawBorderThenFill(top_right_arrow), DrawBorderThenFill(bottom_right_arrow))
        self.play(Write(stephen_no_hat_equation), Write(stephen_hat_equation))
        
        self.embed()
        
        self.play(Uncreate(stephen_hat), Uncreate(stephen_no_hat),
                  FadeOut(arrows),
                  Uncreate())
        
        x_axis = WelchXAxis(-5, 5).move_to(ORIGIN)
        y_axis = WelchYAxis(-5, 5).move_to(ORIGIN)
        x_axis.ticks.set_opacity(0)
        x_axis.labels.set_opacity(0)
        y_axis.ticks.set_opacity(0)
        y_axis.labels.set_opacity(0)
        
        axes = VGroup(x_axis, y_axis)
        axes.shift(LEFT * 4)
        
        
def generate_nice_ticks(min_val, max_val, min_ticks=3, max_ticks=16, ignore=[0]):
    """
    Generate a list of nice-looking tick values between min_val and max_val,
    and return extended range values for the full axis.
    
    Args:
        min_val (float): Minimum value for the data range
        max_val (float): Maximum value for the data range
        min_ticks (int): Minimum number of ticks desired
        max_ticks (int): Maximum number of ticks desired
        ignore (list): List of values to exclude from the ticks
        
    Returns:
        tuple: (tick_values, axis_min, axis_max)
            - tick_values (list): A list of tick values
            - axis_min (float): Suggested minimum value for the axis (one tick before min_val)
            - axis_max (float): Suggested maximum value for the axis (one tick after max_val)
    """
    # Ensure min_val < max_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        
    # Handle case where min_val and max_val are equal or very close
    if abs(max_val - min_val) < 1e-10:
        # Create a small range around the value
        min_val = min_val - 1
        max_val = max_val + 1
    
    # Find the appropriate order of magnitude for the tick spacing
    range_val = max_val - min_val
    power = np.floor(np.log10(range_val))
    
    # Try different multiples of the base power of 10
    possible_step_sizes = [10**power, 5 * 10**(power-1), 2 * 10**(power-1), 10**(power-1)]
    
    # Find the first step size that gives us fewer than max_ticks
    chosen_step = possible_step_sizes[0]  # Default to the largest step
    
    for step in possible_step_sizes:
        # Calculate how many ticks we'd get with this step size
        first_tick = np.ceil(min_val / step) * step
        last_tick = np.floor(max_val / step) * step
        
        # Count ticks, excluding ignored values
        num_ticks = 0
        current = first_tick
        while current <= last_tick * (1 + 1e-10):
            if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
                num_ticks += 1
            current += step
        
        if min_ticks <= num_ticks <= max_ticks:
            chosen_step = step
            break
        elif num_ticks > max_ticks:
            # If we have too many ticks, stop and use the previous step size
            break
    
    # Calculate the first tick at or below min_val
    first_tick = np.floor(min_val / chosen_step) * chosen_step
    
    # Calculate the last tick at or above max_val
    last_tick = np.ceil(max_val / chosen_step) * chosen_step
    
    # Calculate one tick before first_tick for axis_min
    axis_min = first_tick - chosen_step
    
    # Calculate one tick after last_tick for axis_max
    axis_max = last_tick + chosen_step
    
    # Generate the tick values that fall within the data range, excluding ignored values
    ticks = []
    current = np.ceil(min_val / chosen_step) * chosen_step
    
    while current <= max_val * (1 + 1e-10):  # Add a small epsilon to handle floating point errors
        # Only add the tick if it's not in the ignore list
        if not any(abs(current - ignored_val) < 1e-10 for ignored_val in ignore):
            ticks.append(float(current))  # Convert to float to avoid numpy types
        current += chosen_step
    
    # If we still have too few ticks, try the next smaller step size
    if len(ticks) < min_ticks and possible_step_sizes.index(chosen_step) < len(possible_step_sizes) - 1:
        return generate_nice_ticks(min_val, max_val, min_ticks, max_ticks, ignore)
    
    return ticks, float(axis_min), float(axis_max)

class WelchXAxis(VGroup):
    def __init__(
        self,
        x_min=0,
        x_max=6, 
        x_ticks=[1, 2, 3, 4, 5],  # Default tick values
        x_tick_height=0.15,        # Default tick height
        x_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color (using predefined BROWN)
        arrow_tip_scale=0.1, 
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        
        VGroup.__init__(self, **kwargs)

        # Store parameters
        self.x_ticks = x_ticks
        self.x_tick_height = x_tick_height
        self.x_label_font_size = x_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale=arrow_tip_scale
        self.x_min = x_min
        self.x_max = x_max
        self.axis_length_on_canvas=axis_length_on_canvas
        self.include_tip=include_tip

        self.axis_to_canvas_scale=(self.x_max-self.x_min)/axis_length_on_canvas
        self.x_ticks_scaled=(np.array(x_ticks)-self.x_min)/self.axis_to_canvas_scale

        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
        
    def _create_axis_line(self):
        
        # Create a line for the x-axis
        axis_line = Line(
            start=np.array([0, 0, 0]),
            end=np.array([self.axis_length_on_canvas, 0, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        if self.include_tip:
            arrow_tip=SVGMobject('welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([self.axis_length_on_canvas, 0, 0])
            axis_line = VGroup(axis_line, arrow_tip)

        self.axis_line = axis_line
        self.arrow = arrow_tip
        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = VGroup()
        
        for x_val in self.x_ticks_scaled:
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
        
        for x_val, x_val_label in zip(self.x_ticks_scaled, self.x_ticks):
            # In 3B1B's manim, use TexMobject instead of MathTex
            label = Tex(str(round(x_val_label, 4)))
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

    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.x_min)/(self.x_max-self.x_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas

class WelchYAxis(VGroup):
    def __init__(
        self,
        y_min=0,
        y_max=6, 
        y_ticks=[1, 2, 3, 4, 5],  # Default tick values
        y_tick_width=0.15,        # Default tick width
        y_label_font_size=24,     # Default font size
        stroke_width=3,           # Default stroke width
        color=CHILL_BROWN,        # Default color
        arrow_tip_scale=0.1,
        axis_length_on_canvas=5,
        include_tip=True,
        **kwargs
    ):
        VGroup.__init__(self, **kwargs)
        
        # Store parameters
        self.y_ticks = y_ticks
        self.y_tick_width = y_tick_width
        self.y_label_font_size = y_label_font_size
        self.stroke_width = stroke_width
        self.axis_color = color
        self.arrow_tip_scale = arrow_tip_scale
        self.y_min = y_min
        self.y_max = y_max
        self.axis_length_on_canvas = axis_length_on_canvas
        self.include_tip=include_tip
        
        self.axis_to_canvas_scale = (self.y_max - self.y_min) / axis_length_on_canvas
        self.y_ticks_scaled = (np.array(y_ticks)-self.y_min)/ self.axis_to_canvas_scale
        
        # Create the basic components
        self._create_axis_line()
        self._create_ticks()
        self._create_labels()
    def _create_axis_line(self):
        # Create a line for the y-axis
        axis_line = Line(
            start=np.array([0, 0, 0]),
            end=np.array([0, self.axis_length_on_canvas, 0]),
            color=self.axis_color,
            stroke_width=self.stroke_width
        )
        
        # Add SVG arrow tip at the end
        if self.include_tip:
            arrow_tip = SVGMobject('welch_arrow_tip_1.svg')
            arrow_tip.scale(self.arrow_tip_scale)
            arrow_tip.move_to([0, self.axis_length_on_canvas, 0])
            # Rotate the arrow tip to point upward
            arrow_tip.rotate(PI/2)  # Rotate 90 degrees to point up
            axis_line = VGroup(axis_line, arrow_tip)

        self.axis_line = axis_line
        self.add(axis_line)
        self.add(axis_line)
        
    def _create_ticks(self):
        self.ticks = VGroup()
        
        for y_val in self.y_ticks_scaled:
            tick = Line(
                start=np.array([0, y_val, 0]),
                end=np.array([-self.y_tick_width, y_val, 0]),  # Ticks extend to the left
                color=self.axis_color,
                stroke_width=self.stroke_width
            )
            self.ticks.add(tick)
            
        self.add(self.ticks)
        
    def _create_labels(self):
        self.labels = VGroup()
        
        for y_val, y_val_label in zip(self.y_ticks_scaled, self.y_ticks):
            # Use Tex for labels
            label = Tex(str(round(y_val_label,5)))
            label.scale(self.y_label_font_size / 48)  # Approximate scaling
            label.set_color(self.axis_color)
            label.next_to(
                np.array([-self.y_tick_width, y_val, 0]),
                LEFT,
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
    def map_to_canvas(self, value, axis_start=0):
        value_scaled=(value-self.y_min)/(self.y_max-self.y_min)
        return (value_scaled+axis_start)*self.axis_length_on_canvas

class WelchArrow(VGroup):
    pass