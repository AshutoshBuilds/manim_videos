from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

# class NumpyDataSurface(Surface):
#     def __init__(
#         self,
#         surf_data,
#         xy_data,
#         color=BLUE,
#         opacity=0.8,
#         resolution=(126, 126),  # Matching your data dimensions
#         **kwargs
#     ):
#         self.surf_data = surf_data
#         self.x_values = xy_data[0]  # First row contains x values
#         self.y_values = xy_data[1]  # Second row contains y values
        
#         # Extract ranges from data
#         x_min, x_max = np.min(self.x_values), np.max(self.x_values)
#         y_min, y_max = np.min(self.y_values), np.max(self.y_values)
        
#         super().__init__(
#             color=color,
#             opacity=opacity,
#             u_range=(x_min, x_max),
#             v_range=(y_min, y_max),
#             resolution=resolution,
#             **kwargs
#         )
    
#     def uv_func(self, u, v):
#         # Find the closest indices in our data grid
#         u_idx = np.abs(self.x_values - u).argmin()
#         v_idx = np.abs(self.y_values - v).argmin()
        
#         # Get z value from the surface data
#         # If surf_data is a grid where each point corresponds to a combination
#         # of an x-value and a y-value, we need to find the right index
#         try:
#             z = self.surf_data[u_idx, v_idx]
#         except IndexError:
#             # Fallback if indexing doesn't work as expected
#             z = 0
            
#         return np.array([u, v, z])


class P24v1(InteractiveScene):
    def construct(self):

        surf=np.load('_2025/backprop_1/p_24_28_losses_3.npy')
        xy=np.load('_2025/backprop_1/p_24_28_losses_3xy.npy')

        # Create the surface
        axes = ThreeDAxes(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0, 2.5, 0.5],
            height=5,
            width=5,
            depth=2.5,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        
        # Add labels
        x_label = Tex(r'\theta_{1}', font_size=36).set_color(CHILL_BROWN)
        y_label = Tex(r'\theta_{2}', font_size=36).set_color(CHILL_BROWN)
        z_label = Tex('Loss', font_size=36).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, RIGHT)
        y_label.next_to(axes.y_axis, UP)
        z_label.next_to(axes.z_axis, OUT)
        z_label.rotate(90*DEGREES, [1,0,0])

        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = ParametricSurface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )


        # # Create our custom surface
        # surface = NumpyDataSurface(
        #     surf_data=surf,
        #     xy_data=xy,
        #     color=BLUE_C,
        #     opacity=0.7
        # )
        
        ts = TexturedSurface(surface, '_2025/backprop_1/p_24_28_losses_3.png')
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(0.7)

        #i think there's a better way to do this
        offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        
        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label, ts)
        









