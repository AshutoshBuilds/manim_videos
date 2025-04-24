from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

# surf_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.npy')
# xy_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4xy.npy')

loss_curve_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_5/all_execpt_embedding_random_64.npy')
loss_curve_2=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_5/all_execpt_embedding_random_51.npy')
loss_curve_3=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_19.npy')
loss_curve_4=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_7/all_execpt_embedding_pretrained_27.npy')

alphas_1=np.linspace(-2.5, 2.5, 512)
loss_2d_1=np.load('/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/apr_24_3/pretrained_11_111_first_8.npy')

# import matplotlib.pyplot as plt
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1))
# plt.savefig('loss_2d_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()


def param_surface_1(u, v):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        z = loss_2d_1[u_idx, v_idx]
    except IndexError:
        z = 0
    return np.array([u, v, z])


class P39_48(InteractiveScene):
    def construct(self):
        '''
        Not going to lie I'm kinda terrified of this scene - buuuuut if I can pull of 
        some level of what's in my head I do think it will be DOPE. 
        Ok one step at a time here. Starting with one and then 6 2d panels, then being the last two 
        together into 3d very much like I did in the last big scene. 

        Ok making progress here - I'm pretty certain at this point that all these need to be 
        in "3d upright view" for the transition to 3d to work -> let me go ahead and make that change early-ish. 
        '''

        x_axis_1=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_1=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_1 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_1 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, UP, buff=0.08)

        mapped_x_1=x_axis_1.map_to_canvas(loss_curve_1[0,:]) 
        mapped_y_1=y_axis_1.map_to_canvas(loss_curve_1[1,:])

        # np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T.shape

        curve_1=VMobject()         
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_1=VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1, curve_1)
        axes_1.move_to([0, 0, 0])
        axes_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        self.frame.reorient(0, 90, 0, (-0.21, -0.01, -0.02), 8.00)
        self.add(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.wait(0)
        self.play(ShowCreation(curve_1), run_time=5)

        self.wait() #Ok let's try the "expand in the negative direction deal"


        x_axis_2=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_2=WelchYAxis(y_min=8, y_max=16, y_ticks=[8, 9, 10, 11, 12, 13, 14, 15 ], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_2 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_2 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, UP, buff=0.08)

        mapped_x_2=x_axis_2.map_to_canvas(loss_curve_2[0,:]) 
        mapped_y_2=y_axis_2.map_to_canvas(loss_curve_2[1,:])

        curve_2=VMobject()         
        curve_2.set_points_smoothly(np.vstack((mapped_x_2, mapped_y_2, np.zeros_like(mapped_x_2))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=1.0)

        axes_2=VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2, curve_2)
        axes_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        axes_2.move_to([10, 0, 0])

        self.play(self.frame.animate.reorient(0, 90, 0, (5.03, 0.05, 0.0), 12.15), 
                  FadeIn(VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2)),
                  ShowCreation(curve_2),
                  run_time=3.0)
        self.wait()


        x_axis_3=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_3=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_3 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_3 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_3.next_to(x_axis_3, RIGHT, buff=0.05)
        y_label_3.next_to(y_axis_3, UP, buff=0.08)

        mapped_x_3=x_axis_3.map_to_canvas(loss_curve_3[0,:]) 
        mapped_y_3=y_axis_3.map_to_canvas(loss_curve_3[1,:])

        curve_3=VMobject()         
        curve_3.set_points_smoothly(np.vstack((mapped_x_3, mapped_y_3, np.zeros_like(mapped_x_3))).T)
        curve_3.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_3=VGroup(x_axis_3, y_axis_3, x_label_3, y_label_3, curve_3)
        axes_3.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        axes_3.move_to([10, 0, 0])


        x_axis_4=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_4=WelchYAxis(y_min=0, y_max=40, y_ticks=[0, 5, 10, 15, 20, 25, 30, 35], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_4 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_4 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_4.next_to(x_axis_4, RIGHT, buff=0.05)
        y_label_4.next_to(y_axis_4, UP, buff=0.08)

        mapped_x_4=x_axis_4.map_to_canvas(loss_curve_4[0,:]) 
        mapped_y_4=y_axis_4.map_to_canvas(loss_curve_4[1,:])

        curve_4=VMobject()         
        curve_4.set_points_smoothly(np.vstack((mapped_x_4, mapped_y_4, np.zeros_like(mapped_x_3))).T)
        curve_4.set_stroke(width=4, color=BLUE, opacity=1.0)

        axes_4=VGroup(x_axis_4, y_axis_4, x_label_4, y_label_4, curve_4)
        axes_4.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        axes_4.move_to([10, 0, -7])

        self.wait()

        self.play(axes_2.animate.move_to([0, 0, -7]),
                 self.frame.animate.reorient(0, 90, 0, (4.6, -3.5, -3.6), 12.54),
                 run_time=2.0)

        self.play(FadeIn(VGroup(x_axis_3, y_axis_3, x_label_3, y_label_3)),
                  FadeIn(VGroup(x_axis_4, y_axis_4, x_label_4, y_label_4)),
                  ShowCreation(curve_3),
                  ShowCreation(curve_4),
                  run_time=4)

        self.wait()

        ## Ok now axes 5 & 6. 
        slice_1=loss_2d_1[255, :]
        slice_2=loss_2d_1[:, 255]

        x_axis_5=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_5=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_5 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_5 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_5.next_to(x_axis_5, RIGHT, buff=0.05)
        y_label_5.next_to(y_axis_5, UP, buff=0.08)

        mapped_x_5=x_axis_5.map_to_canvas(alphas_1) 
        mapped_y_5=y_axis_5.map_to_canvas(slice_1)

        curve_5=VMobject()         
        curve_5.set_points_smoothly(np.vstack((mapped_x_5, mapped_y_5, np.zeros_like(mapped_x_5))).T)
        curve_5.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_5=VGroup(x_axis_5, y_axis_5, x_label_5, y_label_5, curve_5)
        axes_5.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        axes_5.move_to([20, 0, 0])


        x_axis_6=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[-2.0 ,-1.0, 0, 1.0, 2.0], x_tick_height=0.15,        
                            x_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=8)
        y_axis_6=WelchYAxis(y_min=0, y_max=25, y_ticks=[0, 5, 10, 15, 20], y_tick_width=0.15,        
                          y_label_font_size=32, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=6)

        x_label_6 = Tex(r'\alpha', font_size=36).set_color(CHILL_BROWN)
        y_label_6 = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label_6.next_to(x_axis_6, RIGHT, buff=0.05)
        y_label_6.next_to(y_axis_6, UP, buff=0.08)

        mapped_x_6=x_axis_6.map_to_canvas(alphas_1) 
        mapped_y_6=y_axis_6.map_to_canvas(slice_1)

        curve_6=VMobject()         
        curve_6.set_points_smoothly(np.vstack((mapped_x_6, mapped_y_6, np.zeros_like(mapped_x_6))).T)
        curve_6.set_stroke(width=4, color="#FF00FF", opacity=1.0)

        axes_6=VGroup(x_axis_6, y_axis_6, x_label_6, y_label_6, curve_6)
        axes_6.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        axes_6.move_to([20, 0, -7])


        self.add(axes_5, axes_6)
        self.frame.reorient(0, 89, 0, (9.97, -3.5, -3.39), 14.41)






        self.embed()
        self.wait(20)



class sketch_3d(InteractiveScene):
    def construct(self):
        '''
        Hack on 3d surface a bit before I get there, make sure it can do what I want. 
        '''

        # Create main surface
        surface = ParametricSurface(
            param_surface_1,  
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

    
        ts = TexturedSurface(surface, '/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/hackin/loss_2d_1.png')
        ts.set_shading(0.0, 0.1, 0)

        ts.scale([1,1,0.1])
        ts.move_to([0,0,0])

        # self.add(ts)
        self.add(ts)
        
        #Dare we try gridlines lol?

        num_lines = 64  # Number of gridlines in each direction
        num_points = 512  # Number of points per line
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        for u in u_values:
            points = [param_surface_1(u, v) for v in v_points]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            u_gridlines.add(line)

        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface_1(u, v) for u in u_points]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            v_gridlines.add(line)

        u_gridlines.scale([1,1,0.1]) #[scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        u_gridlines.move_to([0,0,0])
        v_gridlines.scale([1,1,0.1]) #[scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])
        v_gridlines.move_to([0,0,0])     

        self.wait()

        self.frame.reorient(32, 38, 0, (0.12, -0.18, -0.23), 7.45)
        self.play(ShowCreation(u_gridlines), 
                  ShowCreation(v_gridlines), 
                  self.frame.animate.reorient(161, 33, 0, (0.11, -0.12, -0.23), 4.84),
                  run_time=10)
        self.wait()
        self.add(ts)

        # self.add(u_gridlines, v_gridlines)  

        # self.remove(ts)

        self.embed()
        self.wait(20)



        # self.frame.reorient(70, 29, 0, (-4.94, 1.73, 1.26), 15.55)
        
        # pivot_x,scale_x=get_pivot_and_scale(axis_min=x_axis_1.x_min, axis_max=x_axis_1.x_max, 
        #                                 axis_end=x_axis_1.axis_length_on_canvas)
        # pivot_y,scale_y=get_pivot_and_scale(axis_min=y_axis_1.y_min, axis_max=y_axis_1.y_max, 
        #                                 axis_end=y_axis_1.axis_length_on_canvas)
        # ts.scale([scale_x, scale_x, scale_y], about_point=[pivot_x, pivot_x, pivot_y])





