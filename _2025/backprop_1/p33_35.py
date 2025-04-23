from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

surf=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.npy')
xy=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4xy.npy')
grads_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_grads_1_2.npy') 
grads_2=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_grads_2_2.npy') 
xy_grads=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_33_35_xy_2.npy') 


def param_surface(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = surf[u_idx, v_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_scaled(u, v):
    u_idx = np.abs(xy[0] - u).argmin()
    v_idx = np.abs(xy[1] - v).argmin()
    try:
        z = 1.5*surf[u_idx, v_idx] #Scale for slightly more dramatic 3d viz. 
    except IndexError:
        z = 0
    return np.array([u, v, z])


def get_grads(u,v):
    u_idx = np.abs(xy_grads[0] - u).argmin()
    v_idx = np.abs(xy_grads[1] - v).argmin()
    try:
        z1 = grads_1[u_idx, v_idx]
    except IndexError:
        z1 = 0
    try:
        z2 = grads_2[u_idx, v_idx]
    except IndexError:
        z2 = 0
    return np.array([u, v, z1, z2])


def map_to_canvas(value, axis_min, axis_max, axis_end, axis_start=0):
    value_scaled=(value-axis_min)/(axis_max-axis_min)
    return (value_scaled+axis_start)*axis_end

def get_pivot_and_scale(axis_min, axis_max, axis_end):
    '''Above collapses into scaling around a single pivot when axis_start=0'''
    scale = axis_end / (axis_max - axis_min)
    return axis_min, scale

class Dot3D(Sphere):
    def __init__(self, center=ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

#Do 2d numerical-ish gradient descent once and use results in a couple places. 
num_steps=10
learning_rate=1.25
grad_adjustment_factors_1=[0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6] # Not sure why I need these - only applying to viz - 
grad_adjustment_factors_2=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] # maybe I should use for descent too?
descent_points=[] 
arrow_end_points_1=[]
arrow_end_points_2=[]

staring_values=param_surface(0, 0)
descent_points.append(list(staring_values)) #First point

for i in range(1, num_steps):
    g=get_grads(descent_points[i-1][0], descent_points[i-1][1]) 
    step_x_1=learning_rate*abs(g[2])
    step_x_2=learning_rate*abs(g[3])
    new_x_1=descent_points[i-1][0]+step_x_1
    new_x_2=descent_points[i-1][1]+step_x_2
    arrow_end_points_1.append([new_x_1, descent_points[i-1][2]+step_x_1*g[2]*grad_adjustment_factors_1[i], 0])
    arrow_end_points_2.append([new_x_2, descent_points[i-1][2]+step_x_2*g[3]*grad_adjustment_factors_2[i], 0])
    descent_points.append([new_x_1, new_x_2, param_surface(new_x_1, new_x_2)[2]])
    # print(g)
# print(descent_points)
arrow_end_points_1=np.array(arrow_end_points_1)
arrow_end_points_2=np.array(arrow_end_points_2)
descent_points=np.array(descent_points)



class P34v2D(InteractiveScene):
    def construct(self):
        '''
        Let's begin by running 2d numerical-ish gradient descent and visiaulizing it in 2 1d panes
        Curves should update as we move in the 2d space. 
        Ideas where initially developed in p33_35_sketch.py - lots of notes there too. 
        '''

        #Now that grads are computed, start viz. 
        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=VGroup(x_axis_1, y_axis_1)

        x_label_1 = Tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, UP, buff=0.05)

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=VGroup(x_axis_2, y_axis_2)

        x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 


        ## Get all Curves, Points, Arrows, and Lines, then Group into panels
        curves_1=VGroup(); curves_2=VGroup()
        points_1=VGroup(); points_2=VGroup()
        arrows_1=VGroup(); arrows_2=VGroup()
        lines_1=VGroup(); lines_2=VGroup()

        #Map all points
        descent_points_mapped_1=np.zeros_like(descent_points)
        descent_points_mapped_1[:,0]=map_to_canvas(descent_points[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_1[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        descent_points_mapped_2=np.zeros_like(descent_points)
        descent_points_mapped_2[:,0]=map_to_canvas(descent_points[:,1], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_2[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_1_mapped=np.zeros_like(arrow_end_points_1)
        arrow_end_points_1_mapped[:,0]=map_to_canvas(arrow_end_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_1_mapped[:,1]=map_to_canvas(arrow_end_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_2_mapped=np.zeros_like(arrow_end_points_2)
        arrow_end_points_2_mapped[:,0]=map_to_canvas(arrow_end_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_2_mapped[:,1]=map_to_canvas(arrow_end_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        for i in range(len(descent_points)):
            #Curves 1
            p1 = np.array([param_surface(u, descent_points[i][1]) for u in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,0], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=YELLOW, opacity=0.8)
            curves_1.add(c)

            #Points 1
            p=Dot(descent_points_mapped_1[i], radius=0.06, fill_color=YELLOW)
            points_1.add(p)

            # #Curves 2
            p1 = np.array([param_surface(descent_points[i][0], v) for v in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,1], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=BLUE, opacity=0.8)
            curves_2.add(c)

            #Points 2
            p=Dot(descent_points_mapped_2[i], radius=0.06, fill_color=BLUE)
            points_2.add(p)
          
            if i>0:
                lines_1.add(Line(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0], 
                                   end=[descent_points_mapped_1[i][0], descent_points_mapped_1[i][1], 0], 
                                   color=YELLOW, buff=0, stroke_width=1.5))   
                arrows_1.add(Arrow(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0], 
                                     end=arrow_end_points_1_mapped[i-1], fill_color=YELLOW, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))
                lines_2.add(Line(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0], 
                                   end=[descent_points_mapped_2[i][0], descent_points_mapped_2[i][1], 0], 
                                   color=BLUE, buff=0, stroke_width=1.5))   
                arrows_2.add(Arrow(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0], 
                                     end=arrow_end_points_2_mapped[i-1], fill_color=BLUE, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))

        ## Test viz as I go here            
        panel_1=VGroup(axes_1, x_label_1, y_label_2, curves_1, points_1, arrows_1, lines_1)
        panel_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        panel_2=VGroup(axes_2, x_label_2, y_label_2, curves_2, points_2, arrows_2, lines_2)
        panel_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)     

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        self.add(panel_1, panel_2)

        self.frame.reorient(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)
        self.wait()


        #WONDER IF I CAN DO A COOL TRANSFOMR INTO ANIMATION DEAL AS THE CUVES CHANGE
        #
        # Ok, I think all 2d pieces are in place here, after i rewrite these paragraphs I'll come figure out the animatino. 


        self.embed()
        self.wait(20)


class P33_35(InteractiveScene):
    def construct(self):
        '''
        Ok like 100 lines of duplicated code here -> I'm tempted to roll into a shared function, but that seems premature and
        i would lose some variable access - I think i can live with the duplication. 
        '''
        
        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=VGroup(x_axis_1, y_axis_1)

        x_label_1 = Tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, UP, buff=0.05)

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=2.2, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=VGroup(x_axis_2, y_axis_2)

        x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 


        ## Get all Curves, Points, Arrows, and Lines, then Group into panels
        curves_1=VGroup(); curves_2=VGroup()
        points_1=VGroup(); points_2=VGroup()
        arrows_1=VGroup(); arrows_2=VGroup()
        lines_1=VGroup(); lines_2=VGroup()

        #Map all points
        descent_points_mapped_1=np.zeros_like(descent_points)
        descent_points_mapped_1[:,0]=map_to_canvas(descent_points[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_1[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        descent_points_mapped_2=np.zeros_like(descent_points)
        descent_points_mapped_2[:,0]=map_to_canvas(descent_points[:,1], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        descent_points_mapped_2[:,1]=map_to_canvas(descent_points[:,2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_1_mapped=np.zeros_like(arrow_end_points_1)
        arrow_end_points_1_mapped[:,0]=map_to_canvas(arrow_end_points_1[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_1_mapped[:,1]=map_to_canvas(arrow_end_points_1[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        arrow_end_points_2_mapped=np.zeros_like(arrow_end_points_2)
        arrow_end_points_2_mapped[:,0]=map_to_canvas(arrow_end_points_2[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        arrow_end_points_2_mapped[:,1]=map_to_canvas(arrow_end_points_2[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        for i in range(len(descent_points)):
            #Curves 1
            p1 = np.array([param_surface(u, descent_points[i][1]) for u in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,0], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=YELLOW, opacity=0.8)
            curves_1.add(c)

            #Points 1
            p=Dot(descent_points_mapped_1[i], radius=0.06, fill_color=YELLOW)
            points_1.add(p)

            # #Curves 2
            p1 = np.array([param_surface(descent_points[i][0], v) for v in np.linspace(-1, 4, 128)])
            points_mapped=np.zeros_like(p1)
            points_mapped[:,0]=map_to_canvas(p1[:,1], axis_min=x_axis_1.x_min, 
                                             axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
            points_mapped[:,1]=map_to_canvas(p1[:,2], axis_min=y_axis_1.y_min, 
                                             axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
            c = VMobject()
            c.set_points_smoothly(points_mapped)
            c.set_stroke(width=4, color=BLUE, opacity=0.8)
            curves_2.add(c)

            #Points 2
            p=Dot(descent_points_mapped_2[i], radius=0.06, fill_color=BLUE)
            points_2.add(p)
          
            if i>0:
                lines_1.add(Line(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0], 
                                   end=[descent_points_mapped_1[i][0], descent_points_mapped_1[i][1], 0], 
                                   color=YELLOW, buff=0, stroke_width=1.5))   
                arrows_1.add(Arrow(start=[descent_points_mapped_1[i-1][0], descent_points_mapped_1[i-1][1], 0], 
                                     end=arrow_end_points_1_mapped[i-1], fill_color=YELLOW, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))
                lines_2.add(Line(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0], 
                                   end=[descent_points_mapped_2[i][0], descent_points_mapped_2[i][1], 0], 
                                   color=BLUE, buff=0, stroke_width=1.5))   
                arrows_2.add(Arrow(start=[descent_points_mapped_2[i-1][0], descent_points_mapped_2[i-1][1], 0], 
                                     end=arrow_end_points_2_mapped[i-1], fill_color=BLUE, 
                                     thickness=3.0, tip_width_ratio=5, buff=0))

        ## Test viz as I go here            
        panel_1=VGroup(axes_1, x_label_1, y_label_1, curves_1, points_1, arrows_1, lines_1)
        panel_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        panel_2=VGroup(axes_2, x_label_2, y_label_2, curves_2, points_2, arrows_2, lines_2)
        panel_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)     

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        # self.add(panel_1, panel_2)

        self.frame.reorient(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        curves_1[0].set_stroke(opacity=0.5)
        curves_2[0].set_stroke(opacity=0.5)
        panel_1_start=VGroup(axes_1, x_label_1, y_label_1, curves_1[0], points_1[0], arrows_1[0])
        panel_2_start=VGroup(axes_2, x_label_2, y_label_2, curves_2[0], points_2[0], arrows_2[0])

        self.add(panel_1_start)
        self.add(panel_2_start)


        r=panel_2.get_corner(LEFT+BOTTOM) 
        r[0]=-4.15    
        self.wait()

        self.play(y_axis_2.animate.set_opacity(0.0),
                  x_axis_1[-1].animate.set_opacity(0.0), #Tick labels
                  x_axis_2[-1].animate.set_opacity(0.0),
                  y_axis_1[-1].animate.set_opacity(0.0),
                  x_axis_1[-2].animate.set_opacity(0.0), #Ticks
                  x_axis_2[-2].animate.set_opacity(0.0),
                  y_axis_1[-2].animate.set_opacity(0.0),
                  y_label_2.animate.set_opacity(0.0),
                  run_time=1.0)

        self.play(panel_1_start.animate.shift([0, 0, -2.0]),
                  panel_2_start.animate.rotate(90*DEGREES, [0,0,1], about_point=r).shift([0, 0, 2.0]),
                  self.frame.animate.reorient(13, 85, 0, (-1.88, -0.77, 1.56), 5.01),
                  run_time=4)

        self.wait()


        self.embed()
        self.wait(20)
























