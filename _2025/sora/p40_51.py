from manimlib import *
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin
)

from typing import Callable

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


class p44_48(InteractiveScene):
    def construct(self):
        '''
        Ok going to try a "clean break" here on the full spiral!
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

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2, 
                                      opacity_range=(0.1, 0.8), fade_length=15)
        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        random_walk[-1]=np.array([0.08, -0.02])
        random_walk=np.cumsum(random_walk,axis=0) 

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[i]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)


        self.frame.reorient(0, 0, 0, (-0.07, 0.01, 0.0), 7.59)
        self.add(axes, dots)
        # self.add(traced_path)
        # self.add(dot_to_move)
        # self.add(dot_history)
        self.wait()

        #P45, zoom in and fade in walk
        self.play(FadeIn(traced_path), 
                  FadeIn(dot_to_move), 
                  # FadeIn(dot_history)
                  )






        self.wait(20)
        self.embed()




class p40_44(InteractiveScene):
    def construct(self):
        '''
        May want to adopt an actual noise schedule here so we don't that big snap at the end - we'll see. 
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

        self.add(axes)
        # self.wait()

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)

        self.wait()

        # Animate the points appearing
        self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)

        self.wait()

        # Ok, let's zoom in on one point, lower opacity on all other points, 
        # and send it on a random walk
        # I think we overlay the image stuff for p42 in illustrator/premier

        #Example_point_index
        i=75
        dot_to_move=dots[i].copy()
        

        self.wait()
        self.play(dots.animate.set_opacity(0.1), 
                 dot_to_move.animate.scale(1.25), #Make main dot a little bigger!
                 self.frame.animate.reorient(0, 0, 0, (2.92, 1.65, 0.0), 4.19), 
                 run_time=2.0)


        self.wait()

        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2, 
                                      opacity_range=(0.1, 0.8), fade_length=15)
        # traced_path.set_opacity(0.5)
        traced_path.set_fill(opacity=0)
        self.add(traced_path)
        self.add(dot_to_move)

        # Ok let's try generating this random walk with a seed in manim - I may want to 
        # compute elsewhere and cache, we'll see. Also, I think proably ignore noise schedule
        # right now? Not sure if/when it will make sense to add. 

        # for random_seed in range(2000):
        #     np.random.seed(random_seed) #4 is maybe best so far, #52 is ok
        #     random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        #     random_walk[0]=np.array([0.3, 0.18]) #make first step go up and to the right
        #     random_walk=np.cumsum(random_walk,axis=0)
        #     random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        #     random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        #     if random_walk_shifted[-1][0]>1.7 and random_walk_shifted[-1][0]<1.9 and random_walk_shifted[-1][1]>1.1 and random_walk_shifted[-1][1]<1.3:
        #         print(random_seed, random_walk_shifted[-1])

        # 485 [1.82222791 1.19839322 0.        ]                                                                                                                         
        # 509 [1.711807   1.28528277 0.        ]                                                                                                                         
        # 1164 [1.75093007 1.23755551 0.        ]                                                                                                                        
        # 1297 [1.88499977 1.28501196 0.        ]                                                                                                                        
        # 1547 [1.86467848 1.20829298 0.        ]                                                                                                                        


        np.random.seed(485) #485 is nice, 4 is maybe best so far, #52 is ok
        random_walk=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
        random_walk[0]=np.array([0.2, 0.12]) #make first step go up and to the right
        random_walk[-1]=np.array([0.08, -0.02])
        random_walk=np.cumsum(random_walk,axis=0) 
        print(random_walk[0])

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        print(random_walk_shifted[-1])

        #Hmm do I want an arrow or does the point just move with a tail?
        self.wait()
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
        self.add(dot_history[-1])
        self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)


        self.wait()

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.22).set_color(YELLOW_FADE))
            self.add(dot_history[-1])
            self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)

        self.wait()


        # traced_path.remove_last_segment()
        # self.remove(dot_history)
        # self.remove(dot_to_move)
        # self.remove(traved_path)




        # Now zoom out and run all paths -> maybe while zoooming out. 
        # I think looking at variable path opacity (highest closes to end point) will be worth looking at for 
        # the batch option. 

        #Ok so I want to fadd in rest of point and center my plot. Then diffuse everybody!!
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.07, 0.01, 0.0), 7.59), 
                        dots.animate.set_opacity(1.0), 
                        run_time=3.0)
        self.wait()

        random_walks=[]
        np.random.seed(2)
        for i in range(100):
            rw=0.07*np.random.randn(100,2) #I might want to manually make the first step larger/more obvious.
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[i][0], batch[i][1], 0])
            random_walks.append(rw_shifted)

        # maybe we actually totally lost the spiral for this animation? That might make for a more dramatic
        # bringing stuff back together phase

        traced_paths=VGroup()
        for idx, d in enumerate(dots): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            if idx != 75:  # Skip the already traced dot
                tp = CustomTracedPath(
                        d.get_center, 
                        stroke_color=YELLOW, 
                        stroke_width=2,
                        opacity_range=(0.1, 0.5),
                        fade_length=10
                    )
                traced_path.set_fill(opacity=0)
                traced_paths.add(tp)
                # tp.set_opacity(0.2)
                # tp.set_fill(opacity=0)
                # traced_path.add(tp)
        self.add(traced_paths)

        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))    


        start_orientation=[0, 0, 0, (-0.07, 0.01, 0.0), 7.59]
        end_orientation=[0, 0, 0, (0.23, -0.24, 0.0), 14.98]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=100)

        self.wait()
        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))
        for step in range(100):
            self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
                     self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)

            #Kinda hacky but just try removing these after first step for now - that first path is distracting for big animation
            self.remove(dot_history)
            self.remove(dot_to_move)
            self.remove(traced_path)
            self.remove(dots[75])

        self.wait()


        
        for tp in traced_paths: tp.stop_tracing()

        #Now play random walk backwards and zoom back in! Don't forget to remove traced paths as we go backwards
        for step in range(99, -1, -1):
            self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
                     self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
            for tp in traced_paths:
                tp.remove_last_segment()
                if step==99: tp.remove_last_segment() #Bug patch
        self.add(dots[75])


        self.wait()



        # Ok ending cleanly on the simple spiral is probably nice here, I can start a new scene - this will help with cleanup etc.  


        self.wait(20)
        self.embed()