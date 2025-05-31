from manimlib import *
from functools import partial
import numpy as np
import torch
import sys
sys.path.append('_2025/backprop_2')
# from network_pranav_pr_1 import *
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


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


viridis_colormap=plt.get_cmap("viridis")
blues_colormap=plt.get_cmap("Blues")
custom_cmap_tans = mcolors.LinearSegmentedColormap.from_list('custom', ['#000000', '#dfd0b9'], N=256)
custom_cmap_cyan = mcolors.LinearSegmentedColormap.from_list('custom', ['#000000', '#00FFFF'], N=256)

# def get_nueron_color(value, vmin=-2, vmax=2):        
#         value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
#         rgba = custom_cmap_tans(value_clipped) #Would also like to try a monochrome tan option
#         return Color(rgb=rgba[:3])

def get_nueron_color(value, vmax=2.5):        
    '''Uses abs, a little reductive'''
    value_clipped = np.clip(np.abs(value)/vmax, 0, 1)
    rgba = custom_cmap_tans(value_clipped) #Would also like to try a monochrome tan option
    return Color(rgb=rgba[:3])

def get_grad_color(value, vmin=-2, vmax=2):        
    value_clipped = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    rgba = custom_cmap_cyan(value_clipped) #Would also like to try a monochrome tan option
    return Color(rgb=rgba[:3])




def get_mlp(w1, 
            w2,
            neuron_fills=None, #Black if None
            grads_1=None,
            grads_2=None,
            line_weight=1.0, 
            line_opacity=0.5, 
            neuron_stroke_width=2.0, 
            neuron_stroke_color='#948979', 
            line_stroke_color='#948979', 
            connection_display_thresh=1.1,
            grad_display_thresh=0.5):

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
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[0][i]), opacity=1.0)
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
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[1][i]), opacity=1.0)
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
            if neuron_fills is None: 
                neuron.set_fill(color='#000000', opacity=1.0)
            else: 
                neuron.set_fill(color=get_nueron_color(neuron_fills[2][i]), opacity=1.0)
            neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
            output_layer.add(neuron)
            
    # Create connections with edge points
    connections = VGroup()
    grad_conections=VGroup()
    
    # Connect input to hidden layer
    for i, in_neuron in enumerate(input_layer):
        for j, hidden_neuron in enumerate(hidden_layer):
            if np.abs(w1[i, j])<connection_display_thresh: continue
            start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
            line = Line(start_point, end_point)
            line.set_stroke(opacity=np.clip(0.8*(np.abs(w1[i, j])-connection_display_thresh), 0, 1), width=line_weight)
            # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
            line.set_color(line_stroke_color)
            connections.add(line)
            if grads_1 is not None:
                if np.abs(grads_1[i, j])<grad_display_thresh: continue
                line_grad = Line(start_point, end_point)
                line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_1[i, j])-grad_display_thresh), 0, 1), 
                                    width=np.abs(grads_1[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))

                line_grad.set_color(get_grad_color(grads_1[i, j]))
                grad_conections.add(line_grad)

            
    # Connect hidden to output layer
    for i, hidden_neuron in enumerate(hidden_layer):
        for j, out_neuron in enumerate(output_layer):
            if np.abs(w2[i, j])<connection_display_thresh: continue
            start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
            line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
            line.set_stroke(opacity=np.clip(0.8*(np.abs(w2[i, j])-connection_display_thresh), 0, 1), width=line_weight)

            line.set_color(line_stroke_color)
            connections.add(line)
            if grads_2 is not None:
                if np.abs(grads_2[i, j])<grad_display_thresh: continue
                line_grad = Line(start_point, end_point)
                line_grad.set_stroke(opacity=np.clip(0.8*(np.abs(grads_2[i, j])-grad_display_thresh), 0, 1), 
                                    width=np.abs(grads_2[i, j]))
                # line_grad.set_stroke(opacity=0.8, width=2)
                # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))

                line_grad.set_color(get_grad_color(grads_2[i, j]))
                grad_conections.add(line_grad)

                
    return VGroup(connections, grad_conections, input_layer, hidden_layer, output_layer, dots)
    # return VGroup(grad_conections, input_layer, hidden_layer, output_layer, dots)



class AttentionPattern(VMobject):
    def __init__(
        self,
        matrix,
        square_size=0.3,
        min_opacity=0.2,
        max_opacity=1.0,
        stroke_width=1.0,
        stroke_color=CHILL_BROWN,
        colormap=custom_cmap_tans,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.matrix = np.array(matrix)
        self.n_rows, self.n_cols = self.matrix.shape
        self.square_size = square_size
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self._colormap = colormap

        self.build()

    def map_value_to_style(self, val):
        val_clipped = np.clip(val, 0, 1)
        rgba = self._colormap(val_clipped)
        color = Color(rgb=rgba[:3])
        opacity = self.min_opacity + val_clipped * (self.max_opacity - self.min_opacity)
        return {"color": color, "opacity": opacity}

    def build(self):
        self.clear()
        squares = VGroup()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                val = self.matrix[i, j]
                style = self.map_value_to_style(val)

                square = Square(side_length=self.square_size)
                square.set_fill(style["color"], opacity=style["opacity"])
                square.set_stroke(self.stroke_color, width=self.stroke_width)

                pos = RIGHT * j * self.square_size + DOWN * i * self.square_size
                square.move_to(pos)
                squares.add(square)

        squares.move_to(ORIGIN)
        self.add(squares)



class LlamaLearningSketchOne(InteractiveScene):
    def construct(self):

        w1 = np.random.randn(20, 24) 
        w2 = np.random.randn(24, 20)  
        grads_1 = np.random.randn(20, 24) 
        grads_2 = np.random.randn(24, 20) 
        neuron_fills=[np.random.randn(20), np.random.randn(24), np.random.randn(20)]

        # net = Network([W1, W2])
        # self.add(net)



        mlp=get_mlp(w1, w2, neuron_fills, grads_1=grads_1, grads_2=grads_2)
        self.add(mlp)
        self.remove(mlp[3][1]); self.add(mlp[3][1]) #Ok seems like I'm just exploiting a bug, but this fixes layering. 
        self.remove(mlp[2][1]); self.add(mlp[2][1])
        mlp.move_to([-4, 0, 0])

        self.wait()

        #Probably wrap this up.
        # def get_attention_layer() 
        attention_border=RoundedRectangle(width=0.6, height=6.3, corner_radius=0.1)
        attention_border.set_stroke(width=0.5, color=CHILL_BROWN)
        self.add(attention_border)


        attention_patterns=VGroup()
        num_attention_patterns=12
        attention_pattern_spacing=0.51
        for i in range(num_attention_patterns):
            if i==num_attention_patterns//2:
                pass
            else:
                matrix = np.random.rand(6, 6)
                attn_pattern = AttentionPattern(matrix=matrix, square_size=0.075, stroke_width=0.5)
                attn_pattern.move_to([0, num_attention_patterns*attention_pattern_spacing/2 - attention_pattern_spacing*(i+0.5), 0])
                attention_patterns.add(attn_pattern)


        self.add(attention_patterns)

        # for i in range(INPUT_NEURONS):
        #     if i == w1.shape[0]//2:  # Middle position for ellipsis
        #         dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
        #         dot.set_color(neuron_stroke_color)
        #         dots.add(dot)
        #     else:


        self.wait() 



        # Ok, making some progress here - I'm going to need to turn off some portion of the connections I think 
        # for it not be become a wall of fill. I think let's try just not drawing neuronss below some threshold?

        # Alright so I have an MLP layer that I don't hate -> I think at this scale basically just turning off
        # most of the connectiosn is the move. And I think that will look ok/fine. 
        # And obviously I'll turn them on/off based on the actually values -> that will be nice!
        # Ok, now there's a few different fun/intersting options I think. 
        # Could have the circles and and attention patterns fill up left to right for forward prop -> that woudl probably be cool?
        # Could be nice/interesting in monochrome brown opacities, or maybe viridis, we'll seee. 
        # I of course could show forward passes along the synapses, we'll see. 
        # Now backward passes I think are definitely along the synapses. 
        # For extra credit I could use glowdots to actually traverse the gaps -> not sure if this is 
        # visually helpful or not, wil keep that as a maybe option. 
        # Ok one more visual option I think it to bring in the bubbles and/or line colors not all at once
        # Could do min to max or randomly -> but having them kinda bubble in could be cool I think.
        # Man i guess activation patterns are more like activations that weights - I guess that's whey thier called activation patterns lol
        # For the monochrome colormap it might be cool to do that nice light tan as the brightest color. 
        #hmm not sure I like viridis fills -> looks kinda meh.
        # Ok monochrome is a little better, but I think the real problem with both is that you kidna want 
        # most of them/the average to be close to black, ya know? 
        # Can i just add an absolute value? Does that lose too much?
        # Ok yeah the absolute value does help visually a bit i think. 
        # Ok that will take some noodling, but I think it's not terrible. 
        # I think it could be cool/interesting to have the activations in monochrom, and the grads in color. 
        #
        # Ok, again for the grads, viridis is kinda meh -> leaning towards monochrom blues or maybe yellows
        # This opens up the possility of showing grads from different examples in different colors - which could be 
        # pretty cool! 
        # I dont' quite want the standard Blues colormaps, ideally i want a grad of zero to be transparent, not white
        # Let me look into that next. If I can't do it with color I should be able to go from black to my color and use 
        # opacity to acheve something similar
        # Not sure yet if I want to show weights and grads or just grads!
        # Ok cool yeah I'm not hating the cyan on top of the weights!
        # Ok cool let's hack on attention patterns a bit now? I'm a bit fuzzy on how I'm going to handle exports - 
        # will be a good challenge for the morning probably - let me see if I can get some basica attention patterns going -
        # I'll start with what Pranav did, it look pretty nice!




        
        self.wait()







        # self.remove(mlp[2])
        # self.add(mlp[2])
        # self.add(mlp[0]) #Connections

        self.wait()
        self.add()

        

        


        mlp[2][1]


        self.wait()
        self.embed()











# def get_mlp(w1, 
#             w2,
#             line_weight=1.0, 
#             line_opacity=0.5, 
#             neuron_stroke_width=2.0, 
#             neuron_stroke_color='#948979', 
#             line_stroke_color='#948979', 
#             connection_display_thresh=1.1):
#     INPUT_NEURONS = w1.shape[0]
#     HIDDEN_NEURONS = w1.shape[1]
#     OUTPUT_NEURONS = w1.shape[0]
#     NEURON_RADIUS = 0.08
#     LAYER_SPACING = 0.25
#     VERTICAL_SPACING = 0.25
#     DOTS_SCALE=0.5
    
#     # Create layers
#     input_layer = VGroup()
#     hidden_layer = VGroup()
#     output_layer = VGroup()
#     dots = VGroup()
    
#     # Input layer
#     for i in range(INPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(LEFT * LAYER_SPACING + UP * ((INPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             input_layer.add(neuron)
            
#     # Hidden layer
#     for i in range(HIDDEN_NEURONS):
#         if i == w1.shape[1]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(UP * ((HIDDEN_NEURONS//2 - i) * VERTICAL_SPACING))
#             hidden_layer.add(neuron)
            
#     # Output layer
#     for i in range(OUTPUT_NEURONS):
#         if i == w1.shape[0]//2:  # Middle position for ellipsis
#             dot = Tex("...").rotate(PI/2, OUT).scale(DOTS_SCALE).move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             dot.set_color(neuron_stroke_color)
#             dots.add(dot)
#         else:
#             neuron = Circle(radius=NEURON_RADIUS, stroke_color=neuron_stroke_color)
#             neuron.set_stroke(width=neuron_stroke_width)
#             neuron.set_fill(color='#000000', opacity=1.0)
#             neuron.move_to(RIGHT * LAYER_SPACING + UP * ((OUTPUT_NEURONS//2 - i) * VERTICAL_SPACING))
#             output_layer.add(neuron)
            
#     # Create connections with edge points
#     connections = VGroup()
    
#     # Connect input to hidden layer
#     for i, in_neuron in enumerate(input_layer):
#         for j, hidden_neuron in enumerate(hidden_layer):
#             if np.abs(w1[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(in_neuron, hidden_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point)
#             line.set_stroke(opacity=np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1), width=line_weight)
#             # print(np.clip(1.0*(np.abs(w1[i, j])-connection_display_thresh), 0, 1))
#             line.set_color(line_stroke_color)
#             connections.add(line)
            
#     # Connect hidden to output layer
#     for i, hidden_neuron in enumerate(hidden_layer):
#         for j, out_neuron in enumerate(output_layer):
#             if np.abs(w2[i, j])<connection_display_thresh: continue
#             start_point, end_point = get_edge_points(hidden_neuron, out_neuron, NEURON_RADIUS)
#             line = Line(start_point, end_point) #, stroke_opacity=line_opacity, stroke_width=line_weight)
#             line.set_stroke(opacity=np.clip(1.0*(np.abs(w2[i, j])-connection_display_thresh), 0, 1), width=line_weight)

#             line.set_color(line_stroke_color)
#             connections.add(line)

                
#     return VGroup(connections, input_layer, hidden_layer, output_layer, dots)











