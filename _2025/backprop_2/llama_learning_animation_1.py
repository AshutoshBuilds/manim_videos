from manimlib import *
from functools import partial
import numpy as np
import torch
import sys
sys.path.append('_2025/backprop_2')
from network_pranav_pr_1 import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

# Helper function to get edge points between two circles
def get_edge_points(circle1, circle2, neuron_radius):
    # Get direction vector from circle1 to circle2
    direction = circle2.get_center() - circle1.get_center()
    unit_vector = direction / np.linalg.norm(direction)
    
    # Calculate start and end points
    start_point = circle1.get_center() + unit_vector * neuron_radius
    end_point = circle2.get_center() - unit_vector * neuron_radius
    
    return start_point, end_point


def get_mlp(w1, 
            w2,
            line_weight=1.0, 
            line_opacity=0.5, 
            neuron_stroke_width=2.0, 
            neuron_stroke_color='#948979', 
            line_stroke_color='#948979', 
            connection_display_thresh=1.1):
    INPUT_NEURONS = w1.shape[0]
    HIDDEN_NEURONS = w1.shape[1]
    OUTPUT_NEURONS = w1.shape[0]
    NEURON_RADIUS = 0.08
    LAYER_SPACING = 0.25
    VERTICAL_SPACING = 0.25
    DOTS_SCALE=0.5
    
    # Create layers
    input_layer = VGroup()
    hidden_layer = VGroup()
    output_layer = VGroup()
    dots = VGroup()
    
    # Input layer
    for i in range(INPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            neuron.set_fill(color=BLACK, opacity=1.0)
            neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            input_layer.add(neuron)
            
    # Hidden layer
    for i in range(HIDDEN_NEURONS):
        if i == w1.shape[1]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            neuron.set_fill(color=BLACK, opacity=1.0)
            neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
            hidden_layer.add(neuron)
            
    # Output layer
    for i in range(OUTPUT_NEURONS):
        if i == w1.shape[0]//2:  # Middle position for ellipsis
            dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            dot.set_color(neuron_stroke_color)
            dots.add(dot)
        else:
            neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
            neuron.set_stroke(width=neuron_stroke_width)
            neuron.set_fill(color=BLACK, opacity=1.0)
            neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            
    # Create connections with edge points
    connections = VGroup()
    
    # Connect input to hidden layer
    for i, in_neuron in enumerate(input_layer):
        for j, hidden_neuron in enumerate(hidden_layer):
            if np.abs(w1[i, j])<connection_display_thresh: continue
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = Line(start_point, end_point)
            line.set_stroke(opacity=np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1), width=line_weight)
            # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
            line.set_color(line_stroke_color)
            connections.add(line)
            
    # Connect hidden to output layer
    for i, hidden_neuron in enumerate(hidden_layer):
        for j, out_neuron in enumerate(output_layer):
            if np.abs(w2[i, j])<connection_display_thresh: continue
            start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
            line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
            line.set_stroke(opacity=np.clip(1.0*(np.abs(w2[i, j])-connection_display_thresh), 0, 1), width=line_weight)

            line.set_color(line_stroke_color)
            connections.add(line)

                
    return VGroup(connections, input_layer, hidden_layer, output_layer, dots)


class LlamaLearningSketchOne(InteractiveScene):
    def construct(self):

        w1 = np.random.randn(20, 24) 
        w2 = np.random.randn(24, 20)  

        # net = Network([W1, W2])
        # self.add(net)

        mlp=get_mlp(w1, w2)
        self.add(mlp)
        self.remove(mlp[2][1]); self.add(mlp[2][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
        self.remove(mlp[1][1]); self.add(mlp[1][1])

        # Ok, making some progress here - I'm going to need to turn off some portion of the connections I think 
        # for it not be become a wall of fill. I think let's try just not drawing neuronss below some threshold?

        self.wait()





        # self.remove(mlp[2])
        # self.add(mlp[2])
        # self.add(mlp[0]) #Connections

        self.wait()
        self.add()

        

        


        mlp[2][1]


        self.wait()
        self.embed()
















