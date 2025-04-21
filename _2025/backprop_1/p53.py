from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'



class P59v1(InteractiveScene):
    def construct(self):

    	# Create the surface
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 1.0],
            height=5,
            width=5,
            depth=3.5,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        self.add(axes)
        self.embed()

        self.wait()




# Claude's first pass - might make sense to break down scene by scene. 
# Start with just the line fitting to data part panel. 

# from manimlib import *
# import numpy as np

# # Define colors
# CHILL_BROWN = '#948979'
# YELLOW = '#ffd35a'
# BLUE = '#65c8d0'
# ORANGE = '#ff7c4d'
# GREEN = '#7cda5f'

# class GradientDescentOnLinearModel(Scene):
#     def construct(self):
#         # Create synthetic data
#         np.random.seed(42)  # For reproducibility
#         num_points = 20
#         x_data = np.random.uniform(0, 10, num_points)
#         # True parameters: slope=2, intercept=1
#         true_theta1, true_theta2 = 2.0, 1.0
#         y_data = true_theta1 * x_data + true_theta2 + np.random.normal(0, 1, num_points)
        
#         # Create layout with two panels
#         self.create_split_screen(x_data, y_data, true_theta1, true_theta2)
        
#     def create_split_screen(self, x_data, y_data, true_theta1, true_theta2):
#         # Create left panel - 3D loss landscape
#         loss_surface_view = self.create_loss_surface_view(x_data, y_data)
#         loss_surface_view.scale(0.5).to_edge(LEFT)
        
#         # Create right panel - Data fitting view
#         data_fitting_view = self.create_data_fitting_view(x_data, y_data)
#         data_fitting_view.scale(0.5).to_edge(RIGHT)
        
#         # Add both panels to the scene
#         self.add(loss_surface_view, data_fitting_view)
        
#         # Run gradient descent
#         self.animate_gradient_descent(loss_surface_view, data_fitting_view, x_data, y_data)
        
#     def create_loss_surface_view(self, x_data, y_data):
#         # Create a group to hold all 3D objects
#         surface_group = VGroup()
        
#         # Create 3D axes
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 10.0, 2.0],
#             height=6,
#             width=6,
#             depth=4,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add axis labels
#         x_label = Text("θ₁", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("θ₂", font_size=24).next_to(axes.y_axis.get_end(), UP)
#         z_label = Text("Loss", font_size=24).next_to(axes.z_axis.get_end(), OUT)
        
#         axes_labels = VGroup(x_label, y_label, z_label)
        
#         # Create the loss surface
#         surface = self.create_loss_surface(axes, x_data, y_data)
        
#         # Add a title
#         title = Text("Loss Surface", font_size=30).to_edge(UP)
        
#         # Create a parameter point that will move on the surface
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         initial_loss = self.loss_function(initial_theta1, initial_theta2, x_data, y_data)
#         parameter_point = Sphere(radius=0.1, color=ORANGE)
#         parameter_point.move_to(axes.c2p(initial_theta1, initial_theta2, initial_loss))
        
#         # Add elements to the group
#         surface_group.add(axes, axes_labels, surface, parameter_point, title)
        
#         # Set initial camera orientation for better view
#         frame = self.camera.frame
#         frame.set_euler_angles(phi=70 * DEGREES, theta=30 * DEGREES)
        
#         return surface_group
        
#     def create_loss_surface(self, axes, x_data, y_data):
#         # Function to compute z for any (x, y)
#         def param_surface(u, v):
#             theta1 = u
#             theta2 = v
#             loss = self.loss_function(theta1, theta2, x_data, y_data)
#             return axes.c2p(u, v, loss)
        
#         # Create the surface
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(30, 30),
#             checkerboard_colors=[BLUE, BLUE.lighter(0.5)],
#             fill_opacity=0.7
#         )
        
#         return surface
    
#     def create_data_fitting_view(self, x_data, y_data):
#         # Create a group for data fitting visualization
#         fitting_group = VGroup()
        
#         # Create 2D axes
#         axes = Axes(
#             x_range=[0, 12, 2],
#             y_range=[-1, 25, 5],
#             height=6,
#             width=6,
#             axis_config={"include_tip": True, "color": CHILL_BROWN}
#         )
        
#         # Add axis labels
#         x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
#         axes_labels = VGroup(x_label, y_label)
        
#         # Plot data points
#         data_points = VGroup()
#         for i in range(len(x_data)):
#             point = Dot(axes.c2p(x_data[i], y_data[i]), color=YELLOW, radius=0.05)
#             data_points.add(point)
        
#         # Initial line with random parameters
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         line_function = lambda x: initial_theta1 * x + initial_theta2
#         linear_model = axes.get_graph(line_function, x_range=[0, 12], color=GREEN)
        
#         # Add title
#         title = Text("Data Fitting", font_size=30).to_edge(UP)
        
#         # Add all elements to the group
#         fitting_group.add(axes, axes_labels, data_points, linear_model, title)
        
#         return fitting_group
    
#     def animate_gradient_descent(self, loss_view, data_view, x_data, y_data):
#         # Initialize parameters
#         theta1, theta2 = 0.0, 0.0
#         learning_rate = 0.01
#         num_iterations = 50
        
#         # Get references to the elements we'll animate
#         axes_3d = loss_view[0]  # 3D axes
#         parameter_point = loss_view[3]  # Parameter point on loss surface
        
#         axes_2d = data_view[0]  # 2D axes
#         linear_model = data_view[3]  # Line representing the model
        
#         # Save original positions
#         original_3d_pos = parameter_point.get_center()
        
#         # Run gradient descent with animations
#         for i in range(num_iterations):
#             # Compute gradients
#             grad_theta1 = self.gradient_theta1(theta1, theta2, x_data, y_data)
#             grad_theta2 = self.gradient_theta2(theta1, theta2, x_data, y_data)
            
#             # Update parameters
#             theta1 = theta1 - learning_rate * grad_theta1
#             theta2 = theta2 - learning_rate * grad_theta2
            
#             # Compute current loss
#             current_loss = self.loss_function(theta1, theta2, x_data, y_data)
            
#             # Create new target position for parameter point
#             new_position_3d = axes_3d.c2p(theta1, theta2, current_loss)
            
#             # Create new line for the updated model
#             updated_line_function = lambda x: theta1 * x + theta2
#             new_line = axes_2d.get_graph(updated_line_function, x_range=[0, 12], color=GREEN)
            
#             # Animate both simultaneously
#             self.play(
#                 parameter_point.animate.move_to(new_position_3d),
#                 Transform(linear_model, new_line),
#                 run_time=0.2 if i > 10 else 0.5  # Speed up later iterations
#             )
            
#             # Add a brief pause at the beginning to see initial state
#             if i == 0:
#                 self.wait(1)
        
#         # Pause at the end to see final state
#         self.wait(2)
    
#     # Loss function: Mean Squared Error
#     def loss_function(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return np.mean(errors**2)
    
#     # Gradient with respect to theta1
#     def gradient_theta1(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(x_data * errors)
    
#     # Gradient with respect to theta2
#     def gradient_theta2(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(errors)


# class GradientDescentOn3DSurface(ThreeDScene):
#     def construct(self):
#         # Create synthetic data
#         np.random.seed(42)  # For reproducibility
#         num_points = 20
#         x_data = np.random.uniform(0, 10, num_points)
#         # True parameters: slope=2, intercept=1
#         true_theta1, true_theta2 = 2.0, 1.0
#         y_data = true_theta1 * x_data + true_theta2 + np.random.normal(0, 1, num_points)
        
#         # Create 3D axes
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 10.0, 2.0],
#             height=8,
#             width=8,
#             depth=5,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add axis labels
#         x_label = Text("θ₁", font_size=30).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("θ₂", font_size=30).next_to(axes.y_axis.get_end(), UP)
#         z_label = Text("Loss", font_size=30).next_to(axes.z_axis.get_end(), OUT)
        
#         # Set up the 3D scene
#         self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
#         self.add(axes, x_label, y_label, z_label)
        
#         # Create the loss surface
#         surface = self.create_loss_surface(axes, x_data, y_data)
#         self.play(FadeIn(surface))
        
#         # Create a parameter point that will move on the surface
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         initial_loss = self.loss_function(initial_theta1, initial_theta2, x_data, y_data)
#         parameter_point = Sphere(radius=0.15, color=ORANGE)
#         parameter_point.move_to(axes.c2p(initial_theta1, initial_theta2, initial_loss))
#         self.play(FadeIn(parameter_point))
        
#         # Run gradient descent with animations
#         self.animate_gradient_descent(axes, parameter_point, x_data, y_data)
        
#         # Final pause
#         self.wait(2)
        
#     def create_loss_surface(self, axes, x_data, y_data):
#         # Function to compute z for any (x, y)
#         def param_surface(u, v):
#             theta1 = u
#             theta2 = v
#             loss = self.loss_function(theta1, theta2, x_data, y_data)
#             return axes.c2p(u, v, loss)
        
#         # Create the surface
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(30, 30),
#             checkerboard_colors=[BLUE, BLUE.lighter(0.5)],
#             fill_opacity=0.7
#         )
        
#         return surface
    
#     def animate_gradient_descent(self, axes, parameter_point, x_data, y_data):
#         # Initialize parameters
#         theta1, theta2 = 0.0, 0.0
#         learning_rate = 0.01
#         num_iterations = 100
#         path_points = []
        
#         # First iteration outside loop to get starting point
#         current_loss = self.loss_function(theta1, theta2, x_data, y_data)
#         path_points.append(axes.c2p(theta1, theta2, current_loss))
        
#         # Run gradient descent
#         for i in range(num_iterations):
#             # Compute gradients
#             grad_theta1 = self.gradient_theta1(theta1, theta2, x_data, y_data)
#             grad_theta2 = self.gradient_theta2(theta1, theta2, x_data, y_data)
            
#             # Update parameters
#             theta1 = theta1 - learning_rate * grad_theta1
#             theta2 = theta2 - learning_rate * grad_theta2
            
#             # Compute current loss
#             current_loss = self.loss_function(theta1, theta2, x_data, y_data)
            
#             # Save point for path
#             path_points.append(axes.c2p(theta1, theta2, current_loss))
            
#             # Create new target position for parameter point
#             new_position = axes.c2p(theta1, theta2, current_loss)
            
#             # Animate at different speeds depending on iteration
#             if i < 10:
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.5)
#             elif i < 30:
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.3)
#             elif i % 5 == 0:  # Only show every 5th step for later iterations
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.2)
        
#         # Draw the path followed
#         path = VMobject()
#         path.set_points_smoothly(path_points)
#         path.set_color(ORANGE)
#         path.set_stroke(width=3)
#         self.play(ShowCreation(path))
        
#         # Show the final position for longer
#         self.wait(1)
    
#     # Loss function: Mean Squared Error
#     def loss_function(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return np.mean(errors**2)
    
#     # Gradient with respect to theta1
#     def gradient_theta1(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(x_data * errors)
    
#     # Gradient with respect to theta2
#     def gradient_theta2(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(errors)