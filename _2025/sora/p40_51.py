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
    ScheduleLogLinear, samples, Swissroll, ModelMixin
)

from typing import Callable
from tqdm import tqdm

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


def create_noisy_arrow_animation(self, start_point, end_point, target_point, num_steps=100, noise_level=0.1, overshoot_factor=0.3):
    """
    Creates a sequence of arrow end positions that converge from end_point to target_point
    with parameterizable noise and overshoot past the target direction.
    """
    
    # Calculate initial and target directions
    initial_direction = np.array(end_point) - np.array(start_point)
    target_direction = np.array(target_point) - np.array(start_point)
    
    # Calculate the constant length
    arrow_length = np.linalg.norm(initial_direction)
    
    # Calculate angles
    initial_angle = np.arctan2(initial_direction[1], initial_direction[0])
    target_angle = np.arctan2(target_direction[1], target_direction[0])
    
    # Handle angle wrapping (choose the shorter path)
    angle_diff = target_angle - initial_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Create interpolation parameter
    t_values = np.linspace(0, 1, num_steps)
    
    # Generate noise that decreases over time
    np.random.seed(42)
    noise_decay = np.exp(-3 * t_values)
    angle_noise = noise_level * noise_decay * np.random.randn(num_steps)
    
    # Generate overshoot in angle space - this will make it swing past the target angle
    overshoot_frequency = 3.0
    overshoot_decay = np.exp(-2 * t_values)
    overshoot_oscillation = overshoot_factor * overshoot_decay * np.sin(overshoot_frequency * np.pi * t_values)
    
    # The key: let the angle interpolation overshoot past the target
    t_effective = t_values + overshoot_oscillation
    # Ensure final angle is exactly the target
    t_effective[-1] = 1.0
    
    arrow_positions = []
    
    for i, t_eff in enumerate(t_effective):
        # Interpolate angle - this is where the overshoot happens
        current_angle = initial_angle + t_eff * angle_diff
        
        # Add angular noise (but not on final step)
        if i < len(t_effective) - 1:
            current_angle += angle_noise[i]
        
        # Convert back to cartesian coordinates
        end_x = np.array(start_point)[0] + arrow_length * np.cos(current_angle)
        end_y = np.array(start_point)[1] + arrow_length * np.sin(current_angle)
        
        arrow_positions.append([end_x, end_y, 0])
    
    return arrow_positions





class p48_51(InteractiveScene):
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

        dots = VGroup()
        for point in batch:
            # Map the point coordinates to the axes
            screen_point = axes.c2p(point[0], point[1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots.add(dot)
        dots.set_color(YELLOW)
        dots.set_opacity(0.3)

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
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
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)

        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()

        dot_to_move.set_opacity(1.0)

        #Ok let me try to get all the big elements in here
        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
        x0.next_to(dots[i], 0.2*UP)
        dots[i].set_color('#00FFFF').set_opacity(1.0)

        arrow_x100_to_x0 = Arrow(
            start=dot_to_move.get_center(),
            end=dots[i].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x0.set_color('#00FFFF')
        arrow_x100_to_x0.set_opacity(0.6)

        

        arrow_x100_to_x99 = Arrow(
            start=dot_to_move.get_center(),
            end=[4.739921625933185, 2.8708813273028455, 0], #Just pul in from previous paragraph, kinda hacky but meh. ,
            thickness=1.5,
            tip_width_ratio=5, 
            buff=0.04  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x99.set_color(CHILL_BROWN)
        # arrow_x100_to_x99.set_opacity(0.6)


        self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
        self.add(axes, dots, traced_path, dot_to_move)
        self.add(x100,  x0, arrow_x100_to_x0, arrow_x100_to_x99)
        self.wait()

        # Ok so the continuity to think/worry about here is the brown arrow! Now I'm a bit worried about it's angle - hmm 
        # Let's see how it shakes out. 
        # I think first it's Fading everythig except that data and brown line (maybe scale of brown arrow changes)
        # I might beg able to get away with some updates to the brown arrows angle on a zoom out as I add stuff, we'll see. 

        self.play(FadeOut(traced_path), FadeOut(dot_to_move), FadeOut(x100), FadeOut(x0), FadeOut(arrow_x100_to_x0), 
                 dots.animate.set_opacity(1.0).set_color(YELLOW), run_time=1.5)

        # Ok ok ok so I now in need some vector fields. These come from trained models. Do I want to import the model 
        # and sample from it here? Or do I want to exprot the vector fields? 
        # I think it would be nice to fuck with the density etc in manim, so maybe we get a little aggressive and 
        # try to import the full model? 
        # Lets see here....

        


        self.wait(20)
        self.embed()




class p47b(InteractiveScene):
    def construct(self):
        '''
        Alright need to pick up where i left off on p44_47, and get ready for another crazy particle fly by lol
        This is a little nuts - but I do think it conveys the point nicely. 
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
        dots.set_opacity(0.3)

        i=75
        dot_to_move=dots[i].copy()
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
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
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)

        for j in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[j]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)
        traced_path.stop_tracing()

        dot_to_move.set_opacity(1.0)
        self.frame.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69)
        self.add(axes, dots, traced_path, dot_to_move)
        self.wait()


        #Ok let me try to get all the big elements in here
        x100=Tex('x_{100}', font_size=24).set_color(YELLOW)
        x100.next_to(dot_to_move, 0.07*UP+0.001*RIGHT)

        x99=Tex('x_{99}', font_size=24).set_color(CHILL_BROWN)
        x99.next_to(dot_history[-1], 0.1*UP+0.01*RIGHT)
        dot99=Dot(dot_history[-1].get_center(), radius=0.04)
        dot99.set_color(CHILL_BROWN)

        x0=Tex('x_{0}', font_size=24).set_color('#00FFFF')
        x0.next_to(dots[i], 0.2*UP)
        dots[i].set_color('#00FFFF').set_opacity(1.0)

        arrow_x100_to_x0 = Arrow(
            start=dot_to_move.get_center(),
            end=dots[i].get_center(),
            thickness=1,
            tip_width_ratio=5, 
            buff=0.025  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x0.set_color('#00FFFF')
        arrow_x100_to_x0.set_opacity(0.6)

        arrow_x100_to_x99 = Arrow(
            start=dot_to_move.get_center(),
            end=dot_history[-1].get_center(),
            thickness=1.5,
            tip_width_ratio=5, 
            buff=0.04  # Small buffer so arrow doesn't overlap the dots
        )
        arrow_x100_to_x99.set_color(CHILL_BROWN)
        # arrow_x100_to_x99.set_opacity(0.6)


        self.add(x100, x99, dot99, x0, arrow_x100_to_x0, arrow_x100_to_x99)
        self.wait()

        # Alright probably need to tweak how I'm adding stuff etc -> but let's get to the main event though
        # So probably lost the labels and yellow path? Yeah maybe like this:
        self.remove(x99, traced_path)
        self.wait()

        # Ok so i gotta send a bunch of particles, I can probably just use the same exact animation
        # First let me figure out how I wanto to move the brown arrow
        # It needs to feel noisy, but not too noisy, and coverge to exactly the x0 direction, and length needs
        # to stay the same. And i need 100 steps. Let's ask my buddy Claude. 
        noise_level = 0.06  # Adjust this parameter to control noise amount
        overshoot_factor = 2.0  # Adjust this to control how much overshoot occurs
        start_delay=20
        early_end=10
        arrow_end_positions = create_noisy_arrow_animation(
            self, 
            start_point=dot_to_move.get_center()[:2],  # x100 position (2D)
            end_point=dot_history[-1].get_center()[:2],  # x99 position (2D) 
            target_point=dots[i].get_center()[:2],  # x0 position (2D)
            num_steps=100-start_delay-early_end,
            noise_level=noise_level,
            overshoot_factor=overshoot_factor
        )


        # self.wait()
        # for end_pos in arrow_end_positions:
        #     arrow_x100_to_x99.put_start_and_end_on(
        #         dot_to_move.get_center(),
        #         end_pos
        #         # np.concatenate((end_pos, [0]))
        #     )
        #     self.wait(0.05)  # 5 seconds total for 100 steps

        #Ok that works! Now I need this motion to happen while all the points fly by! Let's try the same points as last time. 


        random_walks=[]
        np.random.seed(2)
        for j in tqdm(range(int(2e6))):
            rw=0.07*np.random.randn(100,2)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[j%len(batch)][0], batch[j%len(batch)][1], 0])
            # if rw_shifted[-1][0]>1.7 and rw_shifted[-1][0]<2.2 and rw_shifted[-1][1]>1.1 and rw_shifted[-1][1]<1.4:
            if rw_shifted[-1][0]>2.1 and rw_shifted[-1][1]>1.4:
                random_walks.append(rw_shifted)

        print(len(random_walks))
        # random_walks=random_walks[:100] #Comment out to do all the points.
        print(len(random_walks))

        dots_to_move = VGroup()
        for j in range(len(random_walks)):
            # Map the point coordinates to the axes
            screen_point = axes.c2p(batch[j%len(batch)][0], batch[j%len(batch)][1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move.add(dot)
        dots_to_move.set_color(FRESH_TAN)
        dots_to_move.set_opacity(0.2)


        traced_paths=VGroup()
        for idx, d in enumerate(dots_to_move): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            if idx != 75:  # Skip the already traced dot
                tp = CustomTracedPath(
                        d.get_center, 
                        stroke_color=FRESH_TAN, 
                        stroke_width=2,
                        opacity_range=(0.01, 0.35),
                        fade_length=10
                    )
                traced_path.set_fill(opacity=0)
                traced_paths.add(tp)
        self.add(traced_paths)

        self.wait()
        for step in range(100):
            self.play(*[dots_to_move[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(random_walks))], 
                     # self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
            if step>start_delay:
                arrow_index=np.clip(step-start_delay, 0, len(arrow_end_positions)-1)
                arrow_x100_to_x99.put_start_and_end_on(dot_to_move.get_center(), arrow_end_positions[arrow_index])
        self.wait()

        self.remove(dots_to_move, traced_paths, dot99) #Might be nice to do a fade out

        self.wait(20)
        self.embed()



class p44_47(InteractiveScene):
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
        traced_path = CustomTracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=3.5, 
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
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
        # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)
        # dot_to_move.move_to(axes.c2p(*random_walk_shifted[1]))
        traced_path.update_path(0.1)

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.4).set_color(YELLOW))
            dot_to_move.move_to(axes.c2p(*random_walk_shifted[i]))
            traced_path.update_path(0.1)
            # self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.1, rate_func=linear)


        self.frame.reorient(0, 0, 0, (-0.07, 0.01, 0.0), 7.59)
        self.add(axes, dots)
        # self.add(traced_path)
        # self.add(dot_to_move)
        # self.add(dot_history)
        self.wait()


        traced_path.stop_tracing()


        #P45, zoom in and fade in walk
        self.play(FadeIn(traced_path), 
                  FadeIn(dot_to_move), 
                  # FadeIn(dot_history)
                  )


        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69),
                 dots.animate.set_opacity(0.3), run_time=3.0)
        self.wait()

        # Ok so p45 will probably be all illustrator/premiere overlay 
        # From here then, I need to send a bunch of diffusion paths out
        # I'm hoping that it will be apparent visually that things are moving more left to right than the other way 
        # But i don't know yet
        # I might want to do a bit of a zoom out here -> we'll see. 
        # Could also draw a box around the neighborhood, or search for ways to accentuate the motion or somethhing - not sure 
        # yet. Let me try the naive approach and see how it feels. 
        # Hmm yeah this one is taking some noodling -> I do kinda think that drawing the neighborhood box might be good. 


        #Ok looks like I need to filter on paths that go through this neighborhood
        random_walks=[]
        np.random.seed(2)
        for j in tqdm(range(int(2e6))):
            rw=0.07*np.random.randn(100,2)
            rw[0]=np.array([0,0]) #make be the starting point
            # rw[-1]=np.array([0.08, -0.02])
            rw=np.cumsum(rw,axis=0) 
            rw=np.hstack((rw, np.zeros((len(rw), 1))))
            rw_shifted=rw+np.array([batch[j%len(batch)][0], batch[j%len(batch)][1], 0])
            # if rw_shifted[-1][0]>1.7 and rw_shifted[-1][0]<2.2 and rw_shifted[-1][1]>1.1 and rw_shifted[-1][1]<1.4:
            if rw_shifted[-1][0]>2.1 and rw_shifted[-1][1]>1.4:
                random_walks.append(rw_shifted)

        print(len(random_walks))
        # random_walks=random_walks[:100]
        print(len(random_walks))

        dots_to_move = VGroup()
        for j in range(len(random_walks)):
            # Map the point coordinates to the axes
            screen_point = axes.c2p(batch[j%len(batch)][0], batch[j%len(batch)][1])
            dot = Dot(screen_point, radius=0.04)
            # dot.set_color(YELLOW)
            dots_to_move.add(dot)
        dots_to_move.set_color(FRESH_TAN)
        dots_to_move.set_opacity(0.3)


        traced_paths=VGroup()
        for idx, d in enumerate(dots_to_move): 
            # tp = TracedPath(d.get_center, stroke_color=YELLOW, stroke_width=2)
            if idx != 75:  # Skip the already traced dot
                tp = CustomTracedPath(
                        d.get_center, 
                        stroke_color=FRESH_TAN, 
                        stroke_width=2,
                        opacity_range=(0.02, 0.5),
                        fade_length=10
                    )
                traced_path.set_fill(opacity=0)
                traced_paths.add(tp)
        self.add(traced_paths)


        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))    
        start_orientation=[0, 0, 0, (3.58, 2.57, 0.0), 2.69]
        end_orientation=[0, 0, 0, (4.86, 2.65, 0.0), 3.06]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=100)
        remaining_indices=np.concatenate((np.arange(75), np.arange(76,len(batch))))

        
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (2.74, 1.72, 0.0), 3.99))

        r=RoundedRectangle(1.5, 1.0, 0.05)
        r.set_stroke(color='#00FFFF', width=2)
        r.move_to(dot_to_move)
        self.add(r)

        self.wait()
        for step in range(100):
            self.play(*[dots_to_move[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in range(len(random_walks))], 
                     # self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
        self.wait()
        
        for tp in traced_paths: tp.stop_tracing()

        self.remove(dots_to_move, traced_paths)
        # self.play(FadeOut(dots_to_move), FadeOut(traced_paths), FadeOut(r))
        self.play(FadeOut(r))
        self.wait()

        #Zoom back in
        self.play(self.frame.animate.reorient(0, 0, 0, (3.58, 2.57, 0.0), 2.69), run_time=3)
        self.wait()


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
        #Reverse process works in interactive mode but not when rendering
        # I'll ask claude later or just play the other clip backwards. 
        for step in range(99, -1, -1):
            self.play(*[dots[i].animate.move_to(axes.c2p(*random_walks[i][step])) for i in remaining_indices], 
                     self.frame.animate.reorient(*interp_orientations[step]), 
                     run_time=0.1, rate_func=linear)
            for tp in traced_paths:
                tp.remove_last_segment()
                if step==99: tp.remove_last_segment() #Bug patch
            self.wait(0.1) #Hmm maybe adding a wait here will help???
        self.add(dots[75])


        self.wait()



        # Ok ending cleanly on the simple spiral is probably nice here, I can start a new scene - this will help with cleanup etc.  


        self.wait(20)
        self.embed()




