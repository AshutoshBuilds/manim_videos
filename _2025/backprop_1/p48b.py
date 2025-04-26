from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class p48b(InteractiveScene):
    def construct(self):
        '''
        Render out nice manim numbers to 4 decimal places for the confidence and cross entropy loss, where 
        loss = -ln(confidence)
        '''
        # Data
        confidences = [0.3916, 0.3412, 0.8123, 0.9531]  # Fake data, replace later with real
        
        # Create labels with MathTex for LaTeX style
        conf_label = Tex(r"\text{Confidence:}", color=WHITE, font_size=54)
        loss_label = Tex(r"\text{Loss:}", color=WHITE, font_size=54)
        
        # Position labels on the left side of the screen
        conf_label.shift(UP * 1 + LEFT * 3)
        loss_label.shift(DOWN * 1 + LEFT * 3)
        
        # Initial confidence and loss values with Tex
        conf_value = Tex(f"{confidences[0]:.4f}", color=WHITE, font_size=54)
        loss_value = Tex(f"{-np.log(confidences[0]):.4f}", color=WHITE, font_size=54)
        
        # Position values to the right of their labels
        conf_value.next_to(conf_label, RIGHT, buff=0.5)
        loss_value.next_to(loss_label, RIGHT, buff=0.5)
        
        # Add everything to the scene
        self.add(conf_label, loss_label, conf_value, loss_value)
        self.wait(1)
        
        # Update the values with instant replacement (no animation transitions)
        for i in range(1, len(confidences)):
            # Remove previous values
            self.remove(conf_value, loss_value)
            
            # Create new values
            conf_value = Tex(f"{confidences[i]:.4f}", color=WHITE, font_size=54)
            loss_value = Tex(f"{-np.log(confidences[i]):.4f}", color=WHITE, font_size=54)
            
            # Position them
            conf_value.next_to(conf_label, RIGHT, buff=0.5)
            loss_value.next_to(loss_label, RIGHT, buff=0.5)
            
            # Add new values
            self.add(conf_value, loss_value)
            self.wait(1)
        
        self.wait(2)




# confidences=[0.3916, 0.3412, 0.8123, 0.9531] #Fake data, replace later with real

# class p48b(InteractiveScene):
#     def construct(self):
#         '''
#         Render out nice manim numbers to 4 decimal places for the confidence and cross entropy loss, where 
#         loss = -ln(confidence)
#         '''
#         # Create labels
#         conf_label = Text("Confidence:", color=YELLOW, font_size=54)
#         loss_label = Text("Loss:", color=BLUE, font_size=36)
        
#         # Position labels on the left side of the screen
#         conf_label.shift(UP * 1 + LEFT * 3)
#         loss_label.shift(DOWN * 1 + LEFT * 3)
        
#         # Create decimal numbers with 4 decimal places
#         conf_value = DecimalNumber(
#             confidences[0],
#             num_decimal_places=4,
#             include_sign=False,
#             font_size=36,
#             color=YELLOW
#         )
        
#         loss_value = DecimalNumber(
#             -np.log(confidences[0]),
#             num_decimal_places=4,
#             include_sign=False,
#             font_size=36,
#             color=BLUE
#         )
        
#         # Position values to the right of their labels
#         conf_value.next_to(conf_label, RIGHT, buff=0.5)
#         loss_value.next_to(loss_label, RIGHT, buff=0.5)
        
#         # Add everything to the scene
#         self.play(
#             Write(conf_label),
#             Write(loss_label),
#             FadeIn(conf_value),
#             FadeIn(loss_value)
#         )
#         self.wait(1)
        
#         # Update the values sequentially for each confidence in our list
#         for i in range(1, len(confidences)):
#             self.play(
#                 conf_value.animate.set_value(confidences[i]),
#                 loss_value.animate.set_value(-np.log(confidences[i])),
#                 run_time=1.5
#             )
#             self.wait(1)
        
#         # Example of cycling through the values repeatedly
#         for _ in range(2):  # Cycle through twice
#             for i in range(len(confidences)):
#                 self.play(
#                     conf_value.animate.set_value(confidences[i]),
#                     loss_value.animate.set_value(-np.log(confidences[i])),
#                     run_time=1
#                 )
#                 self.wait(0.5)
        
#         self.wait(2)



