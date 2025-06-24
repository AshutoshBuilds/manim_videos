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


        model=torch.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/sora/hackin/jun_20_1.pt')

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
            sigma_idx = int(np.clip(current_time * 255 / max_time, 0, 255))
            
            try:
                res = model.forward(torch.tensor(coords_array).float(), sigmas[sigma_idx], cond=None)
                return -res.detach().numpy()
            except:
                return np.zeros((len(coords_array), 2))

        # Create a custom VectorField that updates based on the tracker
        class TrackerControlledVectorField(VectorField):
            def __init__(self, time_tracker, **kwargs):
                self.time_tracker = time_tracker
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

        # Create the tracker-controlled vector field
        vector_field = TrackerControlledVectorField(
            time_tracker=time_tracker,
            func=vector_function_with_tracker,
            coordinate_system=extended_axes,
            density=4.0, #hacking here with Grant - more density??? let's try 4 (was 3) can dial back if I need to.
            stroke_width=2,
            stroke_opacity=0.7, #0.7,
            tip_width_ratio=4,
            tip_len_to_width=0.01,
            max_vect_len_to_step_size=0.7,
            color=CHILL_BROWN
        )        
        
        
        self.add(axes, dots)
        self.wait()

        # Ok so I'll need to noodle with a few different starting points - and am tempted ot start not quite at point 100, ya know?
        #Ok yeah so I need to find path I like...
        path_index=9
        dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[-1, path_index, :], [0]))), radius=0.04)
        dot_to_move.set_color(WHITE)
        self.add(dot_to_move)

        #Look at all dot real quick to get a sanity check on sprial fit
        for path_index in range(512):
            dot_to_move=Dot(axes.c2p(*np.concatenate((xt_history[-1, path_index, :], [0]))), radius=0.04)
            dot_to_move.set_color(WHITE)
            self.add(dot_to_move)       

        #Ok fit seems pretty good, but maybe like it could be better?
        #Let me compare to jupyter notebook.  

        path_segments=VGroup()
        for k in range(64):
            segment1 = Line(
                axes.c2p(*[xt_history[k, path_index, 0], xt_history[k, path_index, 1]]), 
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]),
                stroke_width=1.5,
                stroke_color=YELLOW
            )
            segment2 = Line(
                axes.c2p(*[history_pre_noise[k, path_index, 0], history_pre_noise[k, path_index, 1]]), 
                axes.c2p(*[xt_history[k+1, path_index, 0], xt_history[k+1, path_index, 1]]),
                stroke_width=1.5,
                stroke_color=WHITE
            )
            path_segments.add(segment1)
            path_segments.add(segment2)
        self.add(path_segments)


        self.remove(path_segments, dot_to_move)

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

