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



class p53(InteractiveScene):
    def construct(self):
        '''
        Ok ok ok need to do a direct transition from p47b after fading out all the traces etc -> then bring
        in the full vector field - I think this is going to be dope!
        '''
        batch_size=2130
        dataset = Swissroll(np.pi/2, 5*np.pi, 100)
        loader = DataLoader(dataset, batch_size=batch_size)
        batch=next(iter(loader)).numpy()

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
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)
        dots.set_opacity(0.3)


        # model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt') #Trained on 256 levels
        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_24_1.pt') #Trained on 64 levels

        schedule = ScheduleLogLinear(N=64, sigma_min=0.01, sigma_max=1) #N=200
        bound=2.0 #Need to match extended axes bro
        num_heatmap_steps=30
        grid=[]
        for i, x in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
            for j, y in enumerate(np.linspace(-bound, bound, num_heatmap_steps)):
                grid.append([x,y])
        grid=torch.tensor(grid).float()

        gam=1
        mu=0.5 #0.5 is DDPM
        cfg_scale=0.0
        cond=None
        sigmas=schedule.sample_sigmas(64)
        xt_history=[]
        history_pre_noise=[]
        heatmaps=[]
        eps=None
        torch.manual_seed(2)

        with torch.no_grad():
            model.eval();
            xt=torch.randn((batch_size,) + model.input_dims)*sigmas[0] #Scaling by sigma here matters a lot - why is that???
            xt_history.append(xt.numpy())
            
            for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
                eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
                # eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
                sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
                eta = (sig_prev**2 - sig_p**2).sqrt()
                history_pre_noise.append(xt - (sig - sig_p) * eps)
                xt = xt - (sig - sig_p) * eps + eta * model.rand_input(xt.shape[0]).to(xt) #Straight remove adding random noise
                
                xt_history.append(xt.numpy())
                heatmaps.append(model.forward(grid, sig, cond=None))
        xt_history=np.array(xt_history)
        history_pre_noise=np.array(history_pre_noise)


        time_tracker = ValueTracker(0.0)  # Start at time 0

        def vector_function_with_tracker(coords_array):
            """Vector function that uses the ValueTracker for time"""
            current_time = time_tracker.get_value()
            max_time = 8.0  # Map time 0-8 to sigma indices 0-255
            sigma_idx = int(np.clip(current_time * 63 / max_time, 0, 63)) #Needs to be N-1
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))

        # Create a custom VectorField that updates based on the tracker
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

        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=3.0,
            stroke_width=2,
            max_radius=6.0,      # Vectors fade to min_opacity at this distance
            min_opacity=0.15,     # Minimum opacity at max_radius
            max_opacity=0.8,     # Maximum opacity at origin
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )
        

        # Ok so I'll need to noodle with a few different starting points - and am tempted ot start not quite at point 100, ya know?
        #Ok yeah so I need to find path I like...
        path_index=25 #Ok I think i like 25? 3 is my fav so far. path 1 is not too shabby, could work. doesn't land quite on the spiral. 
        dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[0, path_index, :], [0]))), radius=0.06)
        dot_to_move.set_color(WHITE)

        path_segments=VGroup()
        for k in range(64):
            segment1 = Line(
                axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                stroke_width=4.0,
                stroke_color=YELLOW
            )
            segment2 = Line(
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                stroke_width=4.0,
                stroke_color=WHITE, 
            )
            segment2.set_opacity(0.4)
            segment1.set_opacity(0.9)
            path_segments.add(segment1)
            path_segments.add(segment2)
        self.add(path_segments) #Add now for layering. 
        path_segments.set_opacity(0.0)


        self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.25)
        self.add(axes)
        self.wait()
        self.play(ShowCreation(dots),
                  self.frame.animate.reorient(0, 0, 0, (0.00, 0.00, 0.0), 8.0), 
                  run_time=3.0)
        self.wait()

        self.play(self.frame.animate.reorient(0, 0, 0, (-1.54, 2.65, 0.0), 6.16),
                  run_time=3.0,
                  )
        self.add(dot_to_move)
        self.wait()

        a0=Arrow(dot_to_move.get_center(), 
                 dot_to_move.get_center()+np.array([2.5, -3.2, 0]), 
                 thickness=3.5,
                 tip_width_ratio=5)
        a0.set_color(YELLOW)
        self.play(FadeIn(a0))
        self.wait()
        self.play(FadeOut(a0))
        self.wait()

        dot_coords=Tex("("+str(round(xt_history[0, path_index, 0], 1))+', '+str(round(xt_history[0, path_index, 1], 1))+")",
                      font_size=32)
        dot_coords.next_to(dot_to_move, DOWN, buff=0.15)
        self.play(Write(dot_coords))
        self.wait()

        self.play(FadeIn(vector_field))
        self.wait()

        #Arrow here or cool variable opacity trail thin here? 
        # a1=Arrow(axes.c2p(*[xt_history[0, path_index, 0], xt_history[0, path_index, 1]]), 
        #          axes.c2p(*[history_pre_noise[0, path_index, 0], history_pre_noise[0, path_index, 1]]),
        #          thickness=3.5,
        #          tip_width_ratio=5)

        self.remove(dot_coords)
        self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[0, path_index, 0], 
                                                         history_pre_noise[0, path_index, 1]])),
                  ShowCreation(path_segments[0]),
                  path_segments[0].animate.set_opacity(0.8),
                  run_time=2.0)
        self.wait()

        self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[1, path_index, 0], 
                                                         xt_history[1, path_index, 1]])),
                  ShowCreation(path_segments[1]),
                  path_segments[1].animate.set_opacity(0.5),
                  run_time=2.0)
        self.wait()

        # self.play(time_tracker.animate.set_value(8.0*(1.0/64.0)), run_time=0.5) #This move is really small, maybe roll it in and actually mention it a little later?

        #Might be nice to lower opacity on older segements as we go? We'll see. 
        for k in range(1, 64):
            self.play(dot_to_move.animate.move_to(axes.c2p(*[history_pre_noise[k, path_index, 0], 
                                                             history_pre_noise[k, path_index, 1]])),
                      ShowCreation(path_segments[2*k]),
                      path_segments[2*k].animate.set_opacity(0.8),
                      run_time=0.4)
            self.wait(0.1)

            self.play(dot_to_move.animate.move_to(axes.c2p(*[xt_history[k+1, path_index, 0], 
                                                             xt_history[k+1, path_index, 1]])),
                      ShowCreation(path_segments[2*k+1]),
                      path_segments[2*k+1].animate.set_opacity(0.5),
                      run_time=0.4)
            # self.wait(0.1)   
            self.play(time_tracker.animate.set_value(8.0*(k/64.0)), run_time=0.1)
               

        self.wait()

        ## ok ok ok ok now zoom out, reset, add a bunch of particles and animate them all!
        ## Everthing in yellow or just to do rainbow hue vibes?
        ## Maybe try rainbow/hue first?
        ## Would be cool it we "landed on" the right colowheel arrangement on the spiral

        self.play(FadeOut(path_segments), FadeOut(dot_to_move), 
                  FadeOut(vector_field), 
                  self.frame.animate.reorient(0, 0, 0, (0.0, 0.0, 0.0), 10), 
                  run_time=4.0)
        self.wait()






        #Don't forget to update vector field as we go! Might want to add a little line in the script about this.
        #Done!
        

        #Look at all dot real quick to get a sanity check on sprial fit
        # for path_index in range(512):
        #     dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[-1, path_index, :], [0]))), radius=0.04)
        #     dot_to_move.set_color(WHITE)
        #     self.add(dot_to_move)       



        # plt.plot([xt_history[k, j, 0], history_pre_noise[k, j, 0]], [xt_history[k, j, 1], history_pre_noise[k, j, 1]], 'm')
        # plt.plot([history_pre_noise[k, j, 0], xt_history[k+1, j, 0]], [history_pre_noise[k, j, 1], xt_history[k+1, j, 1]], 'c')



        # Ok, I think that lowering sigma max for this example defintely makes sense!
        # However I'm not really landing nicely on the spiral! And I want to
        # I think in need to back to jupyter notebook for a bit and tune - mayb revisit Chenyangs original config
        # I want to say he only did 20 steps?!
        # Ok I'm going to back to writing for the weekend I think
        # Made pretty good progress here - will pick back up on tuning spiral DDPM when I'm back on animation - let's go!








        self.wait(20)
        self.embed()

