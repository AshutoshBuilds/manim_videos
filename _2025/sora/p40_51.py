from manimlib import *
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'

from torch.utils.data import DataLoader
from smalldiffusion import (
    ScheduleLogLinear, samples, Swissroll, ModelMixin
)

class p40_51_sketch(InteractiveScene):
    def construct(self):

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

        traced_path = TracedPath(dot_to_move.get_center, stroke_color=YELLOW, stroke_width=2)
        traced_path.set_opacity(0.5)
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
        random_walk[0]=np.array([0.3, 0.18]) #make first step go up and to the right
        random_walk=np.cumsum(random_walk,axis=0) 
        print(random_walk[0])

        random_walk=np.hstack((random_walk, np.zeros((len(random_walk), 1))))
        random_walk_shifted=random_walk+np.array([batch[i][0], batch[i][1], 0])
        print(random_walk_shifted[-1])

        #Hmm do I want an arrow or does the point just move with a tail?
        self.wait()
        
        dot_history=VGroup()
        dot_history.add(dot_to_move.copy().scale(0.25))
        self.add(dot_history[-1])
        self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[0])), run_time=1.0)


        self.wait()

        for i in range(100):
            dot_history.add(dot_to_move.copy().scale(0.25))
            self.add(dot_history[-1])
            self.play(dot_to_move.animate.move_to(axes.c2p(*random_walk_shifted[i])), run_time=0.2, rate_func=linear)

        self.wait()

        #Now zoom out and run all paths -> maybe while zoooming out. 






        self.wait()



        self.wait(20)
        self.embed()



























