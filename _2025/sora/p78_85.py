from manimlib import *
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin, ScheduleDDPM
)

from typing import Callable
from tqdm import tqdm
import torch
from itertools import pairwise
from torch.utils.data import Dataset

def get_color_wheel_colors(n_colors, saturation=1.0, value=1.0, start_hue=0.0):
    """
    Generate N evenly spaced colors from the color wheel.
    
    Args:
        n_colors: Number of colors to generate
        saturation: Color saturation (0.0 to 1.0)
        value: Color brightness/value (0.0 to 1.0) 
        start_hue: Starting hue position (0.0 to 1.0)
    
    Returns:
        List of Manim-compatible hex color strings
    """
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = (start_hue + i / n_colors) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def manual_camera_interpolation(start_orientation, end_orientation, num_steps):
    """
    Linearly interpolate between two camera orientations.
    
    Parameters:
    - start_orientation: List containing camera parameters with a tuple at index 3
    - end_orientation: List containing camera parameters with a tuple at index 3
    - num_steps: Number of interpolation steps (including start and end)
    
    Returns:
    - List of interpolated orientations
    """
    result = []
    
    for step in range(num_steps):
        # Calculate interpolation factor (0 to 1)
        t = step / (num_steps - 1) if num_steps > 1 else 0
        
        # Create a new orientation for this step
        interpolated = []
        
        for i in range(len(start_orientation)):
            if i == 3:  # Handle the tuple at position 3
                start_tuple = start_orientation[i]
                end_tuple = end_orientation[i]
                
                # Interpolate each element of the tuple
                interpolated_tuple = tuple(
                    start_tuple[j] + t * (end_tuple[j] - start_tuple[j])
                    for j in range(len(start_tuple))
                )
                
                interpolated.append(interpolated_tuple)
            else:  # Handle regular numeric values
                start_val = start_orientation[i]
                end_val = end_orientation[i]
                interpolated_val = start_val + t * (end_val - start_val)
                interpolated.append(interpolated_val)
        
        result.append(interpolated)
    
    return result

class CustomTracedPath(VMobject):
    """
    A custom traced path that supports:
    - Reverse playback with segment removal
    - Variable opacity based on distance from end
    - Manual control over path segments
    """
    def __init__(
        self,
        traced_point_func,
        stroke_width=2.0,
        stroke_color=YELLOW,
        opacity_range=(0.1, 0.8),  # (min_opacity, max_opacity)
        fade_length=20,  # Number of segments to fade over
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traced_point_func = traced_point_func
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.opacity_range = opacity_range
        self.fade_length = fade_length
        
        # Store path segments as individual VMobjects
        self.segments = VGroup()
        self.traced_points = []
        self.is_tracing = True
        
        # Add updater for forward tracing
        self.add_updater(lambda m, dt: m.update_path(dt))
    
    def update_path(self, dt=0):
        """Update path during forward animation"""
        if not self.is_tracing or dt == 0:
            return
            
        point = self.traced_point_func()
        self.traced_points.append(point.copy())
        
        if len(self.traced_points) >= 2:
            # Create a new segment
            segment = Line(
                self.traced_points[-2], 
                self.traced_points[-1],
                stroke_width=self.stroke_width,
                stroke_color=self.stroke_color
            )
            
            # Apply opacity gradient
            self.segments.add(segment)
            self.update_segment_opacities()
            self.add(segment)
    
    def update_segment_opacities(self):
        """Update opacity of all segments based on their position"""
        n_segments = len(self.segments)
        if n_segments == 0:
            return
            
        min_op, max_op = self.opacity_range
        
        for i, segment in enumerate(self.segments):
            if i >= n_segments - self.fade_length:
                # Calculate fade based on distance from end
                fade_progress = (i - (n_segments - self.fade_length)) / self.fade_length
                opacity = min_op + (max_op - min_op) * fade_progress
            else:
                opacity = min_op
            segment.set_opacity(opacity)
    
    def remove_last_segment(self):
        """Remove the last segment (for reverse playback)
        Kinda hacky but just run 2x to fix bug
        """
        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()
            # self.update_segment_opacities()

        if len(self.segments) > 0:
            last_segment = self.segments[-1]
            self.segments.remove(last_segment)
            self.remove(last_segment)
            if len(self.traced_points) > 0:
                self.traced_points.pop()

        self.update_segment_opacities()
    
    def stop_tracing(self):
        """Stop the automatic tracing updater"""
        self.is_tracing = False
    
    def start_tracing(self):
        """Resume automatic tracing"""
        self.is_tracing = True
    
    def get_num_segments(self):
        """Get the current number of segments"""
        return len(self.segments)


# class TrackerControlledVectorField(VectorField):
#     def __init__(self, time_tracker, max_radius=2.0, min_opacity=0.1, max_opacity=0.7, **kwargs):
#         self.time_tracker = time_tracker
#         self.max_radius = max_radius  # Maximum radius for opacity calculation
#         self.min_opacity = min_opacity  # Minimum opacity at max radius
#         self.max_opacity = max_opacity  # Maximum opacity at origin
#         super().__init__(**kwargs)
        
#         # Add updater that triggers when tracker changes
#         self.add_updater(self.update_from_tracker)
    
#     def update_from_tracker(self, mob, dt):
#         """Update vectors when tracker value changes"""
#         # Only update if tracker value has changed significantly
#         current_time = self.time_tracker.get_value()
#         if not hasattr(self, '_last_time') or abs(current_time - self._last_time) > 0.01:
#             self._last_time = current_time
#             self.update_vectors()  # Redraw vectors with new time
#             self.apply_radial_opacity()  # Apply opacity falloff after updating
    
#     def apply_radial_opacity(self):
#         """Apply radial opacity falloff from origin"""
#         # Get the stroke opacities array (this creates it if it doesn't exist)
#         opacities = self.get_stroke_opacities()
        
#         # In ManimGL VectorField, each vector is represented by 8 points
#         # Points 0,2,4,6 are the key points of each vector, with point 0 being the base
#         n_vectors = len(self.sample_points)
        
#         for i in range(n_vectors):
#             # Get the base point of this vector (every 8th point starting from 0)
#             base_point = self.sample_points[i]
            
#             # Calculate distance from origin (assuming origin is at [0,0,0])
#             distance = np.linalg.norm(base_point[:2])  # Only use x,y components
            
#             # Calculate opacity based on distance
#             # Linear falloff: opacity decreases linearly with distance
#             opacity_factor = max(0, 1 - distance / self.max_radius)
#             final_opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * opacity_factor
            
#             # Apply the opacity to all 8 points of this vector (except the last one)
#             start_idx = i * 8
#             end_idx = min(start_idx + 8, len(opacities))
#             opacities[start_idx:end_idx] = final_opacity
        
#         # Make sure the data is marked as changed
#         self.note_changed_data()


class TrackerControlledVectorField(VectorField):
    def __init__(self, time_tracker, max_radius=2.0, min_opacity=0.1, max_opacity=0.7, **kwargs):
        self.time_tracker = time_tracker
        self.max_radius = max_radius  # Maximum radius for opacity calculation
        self.min_opacity = min_opacity  # Minimum opacity at max radius
        self.max_opacity = max_opacity  # Maximum opacity at origin
        super().__init__(**kwargs)
        
        # Add updater that triggers when tracker changes
        self.add_updater(self.update_from_tracker)
    
    def update_from_tracker(self, mob, dt):
        """Update vectors when tracker value changes"""
        # Only update if tracker value has changed significantly
        current_time = self.time_tracker.get_value()
        if not hasattr(self, '_last_time') or abs(current_time - self._last_time) > 0.01:
            self._last_time = current_time
            self.update_vectors()  # Redraw vectors with new time
            self.apply_radial_opacity()  # Apply opacity falloff after updating
    
    def set_opacity_range(self, min_opacity, max_opacity):
        """
        Update the opacity range and immediately apply to existing vectors.
        
        Args:
            min_opacity: New minimum opacity (0.0 to 1.0)
            max_opacity: New maximum opacity (0.0 to 1.0)
        """
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        # Immediately apply the new opacity values
        self.apply_radial_opacity()
        return self
    
    def set_max_radius(self, max_radius):
        """
        Update the maximum radius for opacity calculation.
        
        Args:
            max_radius: New maximum radius value
        """
        self.max_radius = max_radius
        # Immediately apply with new radius
        self.apply_radial_opacity()
        return self
    
    def apply_radial_opacity(self):
        """Apply radial opacity falloff from origin"""
        # Get the stroke opacities array (this creates it if it doesn't exist)
        opacities = self.get_stroke_opacities()
        
        # In ManimGL VectorField, each vector is represented by 8 points
        # Points 0,2,4,6 are the key points of each vector, with point 0 being the base
        n_vectors = len(self.sample_points)
        
        for i in range(n_vectors):
            # Get the base point of this vector (every 8th point starting from 0)
            base_point = self.sample_points[i]
            
            # Calculate distance from origin (assuming origin is at [0,0,0])
            distance = np.linalg.norm(base_point[:2])  # Only use x,y components
            
            # Calculate opacity based on distance
            # Linear falloff: opacity decreases linearly with distance
            opacity_factor = max(0, 1 - distance / self.max_radius)
            final_opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * opacity_factor
            
            # Apply the opacity to all 8 points of this vector (except the last one)
            start_idx = i * 8
            end_idx = min(start_idx + 8, len(opacities))
            opacities[start_idx:end_idx] = final_opacity
        
        # Make sure the data is marked as changed
        self.note_changed_data()


class MultiClassSwissroll(Dataset):
    def __init__(self, tmin, tmax, N, num_classes=10, center=(0,0), scale=1.0):

        self.num_classes = num_classes
        
        t = tmin + torch.linspace(0, 1, N) * tmax
        center = torch.tensor(center).unsqueeze(0)
        spiral_points = center + scale * torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T
        
        # Assign classes based on position along the spiral
        # Divide the parameter range into num_classes segments
        class_boundaries = torch.linspace(tmin, tmax, num_classes + 1)
        classes = torch.zeros(N, dtype=torch.long)
        
        for i in range(N):
            # t[i] is already the actual parameter value we want to use for class assignment
            t_val = t[i]
            # Find which segment t_val falls into (0 to num_classes-1)
            class_idx = min(int((t_val - tmin) / (tmax - tmin) * num_classes), num_classes - 1)
            classes[i] = class_idx
        
        # Store data as list of (point, class) tuples
        self.data = [(spiral_points[i], classes[i].item()) for i in range(N)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_class_colors(self):
        """
        Returns a list of colors evenly sampled from a colorwheel (HSV space).
        """
        import matplotlib.colors as mcolors
        
        # Generate evenly spaced hues around the color wheel
        hues = np.linspace(0, 1, self.num_classes, endpoint=False)
        colors = []
        
        for hue in hues:
            # Convert HSV to RGB (saturation=1, value=1 for vibrant colors)
            rgb = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
            colors.append(rgb)
        
        return colors


class p78_85(InteractiveScene):
    def construct(self):

        '''
        Phew - alright last big scene here - Classifier free guidance lets go!!!

        '''


        dataset = MultiClassSwissroll(np.pi/2, 5*np.pi, 100, num_classes=3)
        colors = dataset.get_class_colors()
        loader = DataLoader(dataset, batch_size=len(dataset)*2, shuffle=True)
        # x, labels = next(iter(loader))
        # x=x.cpu().numpy()

        axes = Axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": True,
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )
        axes.set_opacity(0.8)

        # Create extended axes with SAME center point and proportional scaling
        extended_axes = Axes(
            x_range=[-2.0, 2.0, 0.5],    # Extended range
            y_range=[-2.0, 2.0, 0.5],    # Extended range
            height=7 * (4.0/2.4),        # Scale height proportionally: original_height * (new_range/old_range)
            width=7 * (4.0/2.4),         # Scale width proportionally: original_width * (new_range/old_range)
            axis_config={"stroke_width": 0}  # Make invisible
        )

        # Move extended axes to same position as original axes
        extended_axes.move_to(axes.get_center())


        dots = VGroup()
        labels_array=[]
        for point in dataset.data:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0][0], point[0][1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
            labels_array.append(point[1])
        labels_array=np.array(labels_array)
        dots.set_color(YELLOW)
        dots.set_opacity(0.5)



        self.add(axes)
        self.wait()
        self.play(ShowCreation(dots), run_time=0.5)
        self.wait()

        for i, d in enumerate(dots):
            if labels_array[i]==0: 
                d.set_color('#00FFFF').set_opacity(0.9)
                self.wait(0.1)
        self.wait()

        for i, d in enumerate(dots):
            if labels_array[i]==1: 
                d.set_color('#FF00FF').set_opacity(0.9)
                self.wait(0.1)
        self.wait()

        for i, d in enumerate(dots):
            if labels_array[i]==2: 
                d.set_opacity(0.9)
                self.wait(0.1)
        self.wait()

        # Ok will overlay person/dog/cat images in illustrator
        # Now on to p78 - there's a bunch of ways I could show conditioning, but 
        # I think it makes sense to do something similiar to what I did before
        # Like zoom in on Q1, show forward diffusion process, then indicate model training with 
        # f(x, t), and them move parenthesis out to add an extra variable. 

        i=75
        dot_to_move=dots[i].copy()
        dot_to_move.set_opacity(1.0)
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2.0, 
                                      opacity_range=(0.25, 0.9), fade_length=15)
        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        # random_walk[-1]=np.array([0.15, -0.04])
        random_walk[-1]=np.array([0.19, -0.05])
        random_walk=np.cumsum(random_walk,axis=0) 

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([dataset.data[i][0][0], dataset.data[i][0][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)


        self.add(dot_to_move, traced_path)
        dots[i].set_opacity(0.0) #Remove starting dot for now

        start_orientation=[0, 0, 0, (0.00, 0.00, 0.0), 8.0]
        # end_orientation=[0, 0, 0, (2.92, 1.65, 0.0), 4.19]
        end_orientation=[0, 0, 0, (3.48, 1.88, 0.0), 4.26]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, 100)

        self.wait()
        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            self.frame.reorient(*interp_orientations[j])
            self.wait(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()
        self.wait()

        x100=Tex('x_{100}', font_size=24).set_color(WHITE)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)
        self.add(x100)


        pre_point_coords=dot_to_move.get_center()-np.array([0.76, 0.25, 0])
        a2=Arrow(dot_to_move.get_center(),
                 pre_point_coords,
                 thickness = 2.0,
                 tip_width_ratio= 5, 
                 buff=0.035)
        a2.set_color(WHITE)
        # self.add(a2)

        #Lower trace opacity while I bring in equation label


        eq_2=Tex("f(x_{100}, t)", font_size=24)
        eq_2.set_color(WHITE)
        # eq_2[2:7].set_color(YELLOW)
        eq_2.move_to([5.3, 2.7, 0])

        self.play(traced_path.animate.set_opacity(0.1),
                  FadeIn(a2), FadeIn(eq_2))

        self.wait()

        #Ok now add in cat cless here and we're off to the races
        eq_3=Tex("f(x_{100}, t, cat)", font_size=24)
        eq_3.set_color(WHITE)
        # eq_3[2:7].set_color(YELLOW)
        eq_3[-4:-1].set_color(YELLOW)
        eq_3.move_to(eq_2, aligned_edge=LEFT)

        self.play(ReplacementTransform(eq_2[-1], eq_3[-1]))
        self.play(Write(eq_3[-5:-1]))
        self.wait()

        #Now zoom back out to full view. 
        self.play(self.frame.animate.reorient(0,0,0,(0,0,0), 8), run_time=4)
        self.wait()

        # Hmmm hmm ok, so there isn's really a single vector field I can show - right?
        # maybe I just leave out the vector field for now - and just show the paths of the points. 
        # may want to consider playing the same paths as below - we'll see
        xt_history=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_history_3.npy')
        heatmaps=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_heatmaps_3.npy')


        num_dots_per_class=1 #Crank up for final viz
        colors_by_class={2:YELLOW, 0: '#00FFFF', 1: '#FF00FF'}

        all_traced_paths=VGroup()
        all_dots_to_move=VGroup()
        for class_index in range(xt_history.shape[0]):
            for path_index in range(num_dots_per_class): 
                dot_to_move_2=Dot(axes.c2p(*np.concatenate((xt_history[class_index, 0, path_index, :], [0]))), radius=0.06)
                dot_to_move_2.set_color(colors_by_class[class_index])
                all_dots_to_move.add(dot_to_move_2)

                traced_path_2 = CustomTracedPath(dot_to_move_2.get_center, stroke_color=colors_by_class[class_index], stroke_width=2.0, 
                                              opacity_range=(0.0, 1.0), fade_length=12)
                # traced_path_2.set_opacity(0.5)
                # traced_path_2.set_fill(opacity=0)
                all_traced_paths.add(traced_path_2)
        self.add(all_traced_paths)
        self.wait()

        self.play(dots.animate.set_opacity(0.15), 
                 FadeOut(traced_path),
                 FadeOut(dot_to_move),
                 FadeOut(a2), 
                 FadeOut(x100), 
                 eq_3.animate.set_opacity(0.0), 
                 eq_2.animate.set_opacity(0.0),
                 FadeIn(all_dots_to_move)
                 )
        self.wait()


        for k in range(xt_history.shape[1]):
            #Clunky but meh
            animations=[]
            path_index=0
            for class_index in range(xt_history.shape[0]):
                for j in range(num_dots_per_class): 
                    animations.append(all_dots_to_move[path_index].animate.move_to(axes.c2p(*[xt_history[class_index, k, j, 0], 
                                                                                              xt_history[class_index, k, j, 1]])))
                    path_index+=1
            self.play(*animations, rate_func=linear, run_time=0.1)
        self.wait()

        # Ok at p80 now - I think for the first paragraph of p80, 
        # we do all points in gray, and then just highlight cat points 
        # Use colors here that match what I'll use for two different vector fields I think
        # I think it's going to be gray and yellow, lets try that. 

        self.play(FadeOut(all_dots_to_move), 
                  dots.animate.set_color('#777777').set_opacity(1.0))
        self.wait()

        cat_dots=VGroup()
        for i, d in enumerate(dots):
            if labels_array[i]==2: 
                cat_dots.add(d)
        self.play(cat_dots.animate.set_color(YELLOW))
        self.wait()

        #Ok, now cat picture overlay again I think, probably as yello points are coming in. 

        xt_history=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_history_5.npy')
        heatmaps=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_heatmaps_5.npy')
        heatmaps_u=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_heatmaps_5u.npy')
        heatmaps_c=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/conditioned_heatmaps_5c.npy')
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_27_1.pt', map_location=torch.device('cpu'))

        #Setup conditional vector field! If thngs get funky here, switch to using exported heatmaps instead of model

        bound=2.0
        num_heatmap_steps=64
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()


        time_tracker = ValueTracker(0.0)  # Start at time 0
        schedule = ScheduleLogLinear(N=256, sigma_min=0.01, sigma_max=10) #N=200
        sigmas=schedule.sample_sigmas(256)

        # def vector_function_with_tracker(coords_array):
        #     """Vector function that uses the ValueTracker for time"""
        #     current_time = time_tracker.get_value()
        #     max_time = 8.0  # Map time 0-8 to sigma indices 0-255
        #     sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255)) #Needs to be N-1
        #     res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=torch.tensor(2)) #Hardcode to cat for now
        #     return -res.detach().numpy()

        # Let's try the heatmap version - having trouble with model based version
        # If this sucks, try higher resolution, and if that stucks, try model based version again
        def vector_function_heatmap(coords_array):
            """
            Function that takes an array of coordinates and returns corresponding vectors
            coords_array: shape (N, 2) or (N, 3) - array of [x, y] or [x, y, z] coordinates
            Returns: array of shape (N, 2) with [vx, vy] vectors (z component handled automatically)
            """
            result = np.zeros((len(coords_array), 2))
            
            for i, coord in enumerate(coords_array):
                x, y = coord[0], coord[1]  # Take only x, y coordinates
                
                current_time = time_tracker.get_value()
                max_time = 8.0  # Map time 0-8 to sigma indices 0-255
                sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255)) #Needs to be N-1
                # Find the closest grid point to interpolate from
                distances = np.linalg.norm(grid.numpy() - np.array([x, y]), axis=1)
                closest_idx = np.argmin(distances)
                
                # Get the vector at the closest grid point
                vector = heatmaps_c[0, sigma_idx, closest_idx, :]
                result[i] = vector
            
            return -result #Reverse direction



        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_heatmap,
            coordinate_system=extended_axes,
            density=4.0, #5 gives nice detail, but is maybe a little too much, especially to zoom in on soon? Ok I think i like 4.
            stroke_width=2,
            max_radius=5.5,      # Vectors fade to min_opacity at this distance
            min_opacity=0.1,     # Minimum opacity at max_radius
            max_opacity=0.8,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=YELLOW
        )

        #Should use partial() to not repeat so much...
        def vector_function_heatmap_u(coords_array):
            """
            Function that takes an array of coordinates and returns corresponding vectors
            coords_array: shape (N, 2) or (N, 3) - array of [x, y] or [x, y, z] coordinates
            Returns: array of shape (N, 2) with [vx, vy] vectors (z component handled automatically)
            """
            result = np.zeros((len(coords_array), 2))
            
            for i, coord in enumerate(coords_array):
                x, y = coord[0], coord[1]  # Take only x, y coordinates
                
                current_time = time_tracker.get_value()
                max_time = 8.0  # Map time 0-8 to sigma indices 0-255
                sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255)) #Needs to be N-1
                # Find the closest grid point to interpolate from
                distances = np.linalg.norm(grid.numpy() - np.array([x, y]), axis=1)
                closest_idx = np.argmin(distances)
                
                # Get the vector at the closest grid point
                vector = heatmaps_u[0, sigma_idx, closest_idx, :]
                result[i] = vector
            
            return -result #Reverse direction



        # Create the tracker-controlled vector field
        vector_field_u = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_heatmap_u,
            coordinate_system=extended_axes,
            density=4.0, #5 gives nice detail, but is maybe a little too much, especially to zoom in on soon? Ok I think i like 4.
            stroke_width=2,
            max_radius=5.5,      # Vectors fade to min_opacity at this distance
            min_opacity=0.1,     # Minimum opacity at max_radius
            max_opacity=0.8,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color='#777777'
        )




        # self.add(vector_field)

        # self.play(time_tracker.animate.set_value(8.0), run_time=5)
        # self.play(time_tracker.animate.set_value(0.0), run_time=5)


        path_index=70
        guidance_index=0 #No guidance, cfg_scales=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        dot_to_move_3 = Dot(axes.c2p(*[xt_history[guidance_index, 0, path_index, 0], xt_history[guidance_index, 0, path_index, 1], 0]), 
                            radius=0.07)
        dot_to_move_3.set_color(YELLOW)
        dot_to_move_3.set_opacity(1.0)

        traced_path_3 = CustomTracedPath(dot_to_move_3.get_center, stroke_color=WHITE, stroke_width=5.0, 
                                      opacity_range=(0.4, 0.95), fade_length=64)
        traced_path_3.set_fill(opacity=0)
        self.add(traced_path_3)

        self.wait(0)
        self.play(dots.animate.set_opacity(0.2), axes.animate.set_opacity(0.5), 
                  self.frame.animate.reorient(0, 0, 0, (0.23, 2.08, 0.0), 4.78), run_time=2.0)
        self.add(dot_to_move_3)
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        # COMMENTING OUT THIS ANIMATION WHILE WORKING ON LATER ONES -> SLOW!
        for k in range(xt_history.shape[1]):
            self.play(time_tracker.animate.set_value(8.0*(k/256.0)), 
                      dot_to_move_3.animate.move_to(axes.c2p(*[xt_history[guidance_index, k, path_index, 0], 
                                                               xt_history[guidance_index, k, path_index, 1]])),
                     rate_func=linear, run_time=0.01)
        self.wait()

        #Bring up "cat part of sprial to empasize that the point doesnt make it, and I think fade out vector field. 
        self.play(cat_dots.animate.set_opacity(0.7), 
                  FadeOut(vector_field))
        self.wait()


        # Ok hitting p81 now. i think zoom out to overall scene i think? 
        # Many maybe drop vector field? Let's see here...
        # Ok thinking of the storyboard as I go here, but I think there's some good options
        # I think I can introduce the two different vector fields with the f(...) notation I've been using
        
        self.play(self.frame.animate.reorient(0, 0, 0, (0, 0, 0.0), 7.40), run_time=4)
        self.wait()

        self.play(FadeOut(dot_to_move_3), FadeOut(traced_path_3))
        self.wait()

        #A little torn, but I think we go ahead and bring in both vector fields in 81 instead of waiting until 82. 
        #Eh these are pretty overwhelming - let's try just starting with clean f(notation)

        eq_4=Tex("f(x, t)", font_size=48)
        eq_4.set_color(WHITE)
        eq_4.move_to([-4.5, 2, 0])
        eq_4_label=MarkupText("UNCONDITIONAL MODEL", font_size=16, font='myriad-pro')
        eq_4_label.next_to(eq_4, DOWN, buff=0.15).set_color(CHILL_BROWN)
        self.play(Write(eq_4))
        self.play(FadeIn(eq_4_label))
        self.wait()


        eq_5=Tex("f(x, t, cat)", font_size=48)
        eq_5.set_color(WHITE)
        eq_5[-4:-1].set_color(YELLOW)
        eq_5.move_to([4.0, 2, 0])
        eq_5_label=MarkupText("CONDITIONAL MODEL", font_size=16, font='myriad-pro')
        eq_5_label.next_to(eq_5, DOWN, buff=0.15).set_color(CHILL_BROWN)
        self.play(Write(eq_5))
        self.play(FadeIn(eq_5_label))
        self.wait()

        eq_6=Tex("f(x, t, no \  class)", font_size=48)
        eq_6.set_color(WHITE)
        eq_6.move_to(eq_4) #, aligned_edge=LEFT)

        self.play(ReplacementTransform(eq_4[-1], eq_6[-1]), 
                  ReplacementTransform(eq_4[:5], eq_6[:5]))
        self.play(Write(eq_6[-9:-1]))
        self.wait()


        # time_tracker.set_value(3.2)
        self.play(time_tracker.animate.set_value(3.2), run_time=0.1) 
        #This doesnt seem to be taking hmmm - maybe b/c it's faded out?
        #Nuclear option here would be to break to a new scene at 80 -> that probably wouldn't be terrible. 
        # set_value(3.2)

        self.wait()
        self.play(FadeOut(eq_5), FadeOut(eq_5_label))
        self.play(FadeIn(vector_field_u))
        self.wait()

        self.play(eq_4.animate.set_opacity(0.0),
                  eq_6.animate.set_opacity(0.0), 
                  eq_4_label.animate.set_opacity(0.0), 
                  FadeIn(vector_field), 
                  FadeIn(eq_5), 
                  FadeIn(eq_5_label))
        self.wait()

        ## Now time animation - need ot bring diffusion time counter like I did before. 
        time_value = ValueTracker((8-3.2)/8) 
        time_display = DecimalNumber(
            1.0,
            num_decimal_places=2,
            font_size=35,
            color=CHILL_BROWN
        )
        time_display.move_to([-5.4, -3.3, 0]) 
        time_label = MarkupText("t =", font_size=35)
        time_label.set_color(CHILL_BROWN)
        time_label.next_to(time_display, LEFT, buff=0.15)

        # Add updater to keep the display synchronized with the tracker
        time_display.add_updater(lambda m: m.set_value(time_value.get_value()))

        self.play(FadeIn(time_display), FadeIn(time_label))
        self.wait()

        self.play(
            time_tracker.animate.set_value(0.0),  
            time_value.animate.set_value(1.0),    
            run_time=10.0, 
            rate_func=linear
        )
        self.wait()


        self.play(
            time_tracker.animate.set_value(8.0),  
            time_value.animate.set_value(0.0),    
            run_time=10.0, 
            rate_func=linear
        )
        self.wait()        

        # Paragraph 83
        # Ok assuming time updates work for starting config - that gets us to 83!
        # If time update doesn't work I can split this into 2 scenes. 
        # Aright now, for 83 -> i see a few pairs of vectors that I think would be good/fine to show cfg on
        # I want to zoom way in on a single pair, make all other vectors very low opacity or just totally gone
        # and show the geometry of diffisuion guidance, introducing new green vector. 
        # This is going to be cool!
        # Now I think this probably going to take some manual algiment, so I'm thinking 
        # it make sense to skip 83 for a minute, and go ahead and make sure that everything comes together like I 
        # expect/need in 84. 
        # Once I've confirmed that green arrows and path look good, then I'll come back to 83

        # self.frame.reorient(0, 0, 0, (1.41, 1.13, 0.0), 1.05)

        yellow_vec_start=np.array([1.46,1.095,0])
        yellow_vec_vals=np.array([-0.01, 0.15, 0])
        example_vec_yellow=Arrow(yellow_vec_start, 
                                 yellow_vec_start+yellow_vec_vals,
                                 thickness = 0.8,
                                 tip_width_ratio= 5, 
                                 buff=0.0)
        example_vec_yellow.set_color(YELLOW)

        gray_vec_vals=np.array([-0.12, 0.005, 0])
        example_vec_gray=Arrow(yellow_vec_start, 
                                 yellow_vec_start+gray_vec_vals,
                                 thickness = 0.8,
                                 tip_width_ratio= 5, 
                                 buff=0.0)
        example_vec_gray.set_color('#777777')

        green_vec_vals=(yellow_vec_vals-gray_vec_vals) #Fudging the guidance value a bit here. 
        example_vec_green=Arrow(yellow_vec_start+gray_vec_vals, 
                                 yellow_vec_start+gray_vec_vals+green_vec_vals,
                                 thickness = 0.8,
                                 tip_width_ratio= 5, 
                                 buff=0.0)
        example_vec_green.set_color(GREEN)

        green_vec_vals_final=1.8*(yellow_vec_vals-gray_vec_vals) #Fudging the guidance value a bit here. 
        final_vec_green=Arrow(yellow_vec_start+gray_vec_vals, 
                                 yellow_vec_start+gray_vec_vals+green_vec_vals_final,
                                 thickness = 0.8,
                                 tip_width_ratio= 5, 
                                 buff=0.0)
        final_vec_green.set_color(GREEN)

        # green_vec_vals_final_final_lol=1.8*(yellow_vec_vals-gray_vec_vals) #Fudging the guidance value a bit here. 
        final_final_vec_green_lol=Arrow(yellow_vec_start, 
                                 yellow_vec_start+gray_vec_vals+green_vec_vals_final,
                                 thickness = 0.8,
                                 tip_width_ratio= 5, 
                                 buff=0.0)
        final_final_vec_green_lol.set_color(GREEN)


        # self.add(example_vec_yellow, example_vec_gray, final_vec_green)
        #Ok I think this picture is pretty consistent. Sweet.
        # self.remove(example_vec_yellow, example_vec_gray, final_vec_green)
        self.wait()

        #now can i fade in and replace smoothly?

        # eq_5.scale(0.2)
        # eq_5.next_to(example_vec_yellow, RIGHT, buff=0.01)

        # self.add(example_vec_yellow, example_vec_gray) #Add before move? You barley notice
        self.play(FadeOut(time_display), FadeOut(time_label), #FadeOut(eq_5), 
                  FadeOut(eq_5_label),
                  FadeIn(example_vec_yellow), FadeIn(example_vec_gray))
        self.play(self.frame.animate.reorient(0, 0, 0, (1.41, 1.13, 0.0), 1.05), 
                  FadeOut(vector_field_u), 
                  FadeOut(vector_field),
                  eq_5.animate.scale(0.16).next_to(example_vec_yellow, RIGHT, buff=0.015).set_color(YELLOW),
                  run_time=5.0)   
        self.wait()


        eq_7=Tex("f(x, t)", font_size=48)
        eq_7.set_color('#777777').scale(0.16)
        eq_7.next_to(example_vec_gray, DOWN, buff=0.015)
        self.play(FadeIn(eq_7))
        self.wait()

        eq_8=Tex("f(x, t, cat) - f(x,t)", font_size=48)
        eq_8.set_color(GREEN).scale(0.16)
        # eq_8[:len(eq_5)].set_color(YELLOW)
        # eq_8[-len(eq_7):].set_color('#777777')
        eq_5_copy=eq_5.copy()
        eq_7_copy=eq_7.copy()
        eq_8.next_to(eq_5, LEFT, buff=0.17).shift([0,0.02,0])

        # self.add(eq_8)
        # If we take our yellow conditioned vector..
        self.play(ReplacementTransform(eq_5_copy, eq_8[:len(eq_5)]), run_time=2)
        self.wait()
        self.play(ReplacementTransform(eq_7_copy, eq_8[-len(eq_7):]), run_time=2)
        self.add(eq_8); self.remove(eq_5_copy, eq_7_copy)
        self.wait()

        self.play(GrowArrow(example_vec_green))
        self.wait()

        # Ok I think we do a quick Cats/not cats overlay in illustrator to remind folks that yellwow dots are cats. 
        # Cool - done

        eq_9=Tex(r"\alpha (f(x, t, cat) - f(x,t))", font_size=48)
        eq_9.set_color(WHITE).scale(0.16)
        eq_9[2:-1].set_color(GREEN)
        eq_9.move_to(eq_8, aligned_edge=RIGHT).shift([0.05, 0.05, 0])
        self.wait()

        self.play(ReplacementTransform(example_vec_green, final_vec_green), 
                  ReplacementTransform(eq_8, eq_9[2:-1]))
        self.add(eq_9); self.remove(eq_8)
        self.wait()

        #"and replace our original conditioned yellow vector with a vector pointing in this new direction."
        # Ok yeah so I think yellow and maybe green get lower opacity and we add in final final green vector
        self.play(FadeIn(final_final_vec_green_lol), 
                  final_vec_green.animate.set_opacity(0.1),
                  example_vec_yellow.animate.set_opacity(0.1),
                  eq_5.animate.set_opacity(0.1), 
                  eq_9.animate.next_to(final_final_vec_green_lol, LEFT, buff=0.05, aligned_edge=RIGHT))
        self.wait()

        ## Ok ok ok ok now in need a zoom out and smooth transition back to all 3 vector fields!


        # self.play(ApplyWave(cat_dots))
        # self.play(cat_dots.animate.set_opacity(0.9))
        # self.wait()
        # self.play()
        # self.remove(eq_8)


        # Paragraph 84
        # self.wait()

        def vector_function_heatmap_g(coords_array):
            """
            Function that takes an array of coordinates and returns corresponding vectors
            coords_array: shape (N, 2) or (N, 3) - array of [x, y] or [x, y, z] coordinates
            Returns: array of shape (N, 2) with [vx, vy] vectors (z component handled automatically)
            """
            result = np.zeros((len(coords_array), 2))
            
            for i, coord in enumerate(coords_array):
                x, y = coord[0], coord[1]  # Take only x, y coordinates
                
                current_time = time_tracker.get_value()
                max_time = 8.0  # Map time 0-8 to sigma indices 0-255
                sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255)) #Needs to be N-1
                # Find the closest grid point to interpolate from
                distances = np.linalg.norm(grid.numpy() - np.array([x, y]), axis=1)
                closest_idx = np.argmin(distances)
                
                # Get the vector at the closest grid point
                vector = heatmaps[3, sigma_idx, closest_idx, :] #cfg_scales=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
                result[i] = vector
            
            return -result #Reverse direction


        # Create the tracker-controlled vector field
        vector_field_g = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_heatmap_g,
            coordinate_system=extended_axes,
            density=4.0, #5 gives nice detail, but is maybe a little too much, especially to zoom in on soon? Ok I think i like 4.
            stroke_width=2,
            max_radius=5.5,      # Vectors fade to min_opacity at this distance
            min_opacity=0.25,     # Minimum opacity at max_radius
            max_opacity=0.85,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=GREEN
        )

        # self.add(vector_field_g)
        # self.remove(vector_field_g)


        # vector_field

        vector_field.set_opacity_range(0.05, 0.4)
        vector_field_u.set_opacity_range(0.05, 0.4)
        axes.set_opacity(0.4)

        self.wait()
        self.play(FadeIn(vector_field), FadeIn(vector_field_u), FadeIn(vector_field_g), 
                  FadeOut(eq_9), FadeOut(eq_7), FadeOut(eq_5), FadeOut(example_vec_yellow), FadeOut(final_vec_green), 
                  FadeOut(final_final_vec_green_lol), FadeOut(example_vec_gray), cat_dots.animate.set_opacity(0.4),
                  self.frame.animate.reorient(0, 0, 0, (0.23, 2.08, 0.0), 4.78), run_time=5) #Maybe too wide, we'll see
        self.wait()

        #Paragraph 84
        time_display.scale(0.6)
        time_label.scale(0.6)
        time_display.move_to([-3.3,-0.1,0])
        time_label.next_to(time_display, LEFT, buff=0.07)
        self.play(FadeIn(dot_to_move_3), FadeIn(traced_path_3), FadeIn(time_display), FadeIn(time_label))
        self.wait()


        path_index=70
        guidance_index=3 #No guidance, cfg_scales=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        dot_to_move_4 = Dot(axes.c2p(*[xt_history[guidance_index, 0, path_index, 0], xt_history[guidance_index, 0, path_index, 1], 0]), 
                            radius=0.07)
        dot_to_move_4.set_color(GREEN)
        dot_to_move_4.set_opacity(1.0)

        traced_path_4 = CustomTracedPath(dot_to_move_4.get_center, stroke_color=GREEN, stroke_width=5.0, 
                                      opacity_range=(0.6, 0.95), fade_length=64)
        traced_path_4.set_fill(opacity=0)
        self.add(traced_path_4)

        self.wait()


        time_value.set_value((8-3.2)/8)
        #Ok i think i probably want my diffusion time tracker for these last two moves right?
        self.play(time_tracker.animate.set_value(0.0), 
                  time_value.animate.set_value(1.0), run_time=4.0) #Back to=0 here. Can edit this out if I need to
        self.wait()
        self.add(dot_to_move_4)
        self.wait()


        for k in range(xt_history.shape[1]):
            self.play(time_tracker.animate.set_value(8.0*(k/256.0)), 
                      time_value.animate.set_value(1.0-k/256.0),
                      dot_to_move_4.animate.move_to(axes.c2p(*[xt_history[guidance_index, k, path_index, 0], 
                                                               xt_history[guidance_index, k, path_index, 1]])),
                     rate_func=linear, run_time=0.01)
        self.wait()

        # self.add(dot_to_move_3, traced_path_3)

        ## Alright phew this is quite the scene. 
        ## Last two things I think are: 
        ## 1. Zoom out, clear paths, run process with all 3 arrows a bunch of points
        ## 2. Add nice label on top of this legit/cool viz!
        ## Hmm actually let me switch the order - realizing I really need to show the final/full thing in 
        ## one run per class - since the vector field changes. 



        self.wait(20)
        self.embed()






























