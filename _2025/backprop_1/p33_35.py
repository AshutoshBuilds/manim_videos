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

class P33v1(InteractiveScene):
    def construct(self):

        # Ok ok ok ok I need ot decide how automated vs manual I'm going to make this scene
        # Thinking about this scene more, I do think the "right triangle overhead view" of the gradient is going to be nice/important.
        #  
        # Ok ok ok ok ok I do think i want some more Meaat on p33-p35 → specifically it’s interesting and non-obvious that we can put 
        # each slope together in that overhead view and it will point us downhill. Worth thinking about that animation a bit I think → 
        # maybe an actually good use case for WelchAxes lol. And i can draw a connection to part 2 → “as we’ll see in part 2 it turns 
        # out that we can very efficiently estimate the slope of the curves without actually computing an points on them → and then 
        # maybe the little arrows on the curves move to the overhead view? or copies of them? Them more i think about this the more 
        # I think it should be manimy? I can draw the projections/cuts as nice blue/yellow lines in the overhead view too. 
        #
        # Alright kinda low on brain power here, but let me at least try to get the pieces together tonight
        # I do think it probably makes sense to at least try to get all 3 graphs on one canvas, so I can like smoothlly move arrows arround and stuff
        # I might need to go compute a bunch of gradients eh?


        x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_1=VGroup(x_axis_1, y_axis_1)

        points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
        points_mapped=np.array(points_1)[:, (0,2,1)]
        points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        curve_1 = VMobject()
        curve_1.set_points_smoothly(points_mapped)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        x_label_1 = Tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
        y_label_1 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, UP, buff=0.05)

        # self.add(axes_1, curve_1)
        # self.add(x_label_1, y_label_1)

        # Ok, adding second curve should be pretty straightfoward, maybe I think about gradient a bit now?
        # Intuitively I think I need to go actually compuate all the gradients at each point and cache them? 
        # Let me go take a crack at that. 
        # Man I guess you could show the whole gradient field with this approach - that's kinda interesting
        # Not sure if that has a role in the video -> could be fun in the book!

        # I've created a crappy version of c2p I think! Should be able to roll in or absorb. 
        # Keep roling for now though, I can make it pretty after I decide if I like the pattern.

        #Hmm well this is complicated and not working lol - will pick back up in the morning. 
        p1_values=param_surface(0, 0)
        p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
        p1_values[2]=0

        p1=Dot(p1_values, radius=0.06, fill_color=YELLOW)
        # self.add(p1)

        g=get_grads(0,0)
        grad_viz_scale=abs(g[2]) #Maybe make arrow length proportional to gradient?
        p1_values_2=param_surface(0, 0)
        g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
                           [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
                                         axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
        g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
                                         axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

        a1=Arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
        # self.add(a1)

        # panel_1_shift=[-5, 0.4, 0]
        panel_1=VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)
        # panel_1.shift(panel_1_shift)

        # curve_1.set_stroke(opacity=0.25) #Yeah so I a fade out/fade in will bascially be the optning animation for this scene?
        # self.add(panel_1)

        # Ok I might clean up/rewrite this scene, we'll see. 
        # I think it makes sense to to ahead and add panels 2 and 3, and then I can start thinking about how the
        # pieces fit together, and exactly what I want to show from there
        # This workflow feels a bit hacky/uncomfortable, but I do think I have a pretty good feeling of what I want
        # to show, so I'm just going to trust this new/different process. 

        x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
                            x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        axes_2=VGroup(x_axis_2, y_axis_2)


        points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
        points_mapped_2=np.array(points_2)[:, (1,2,0)]
        points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        curve_2 = VMobject()
        curve_2.set_points_smoothly(points_mapped_2)
        curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


        x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
        y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 

        p2_values=param_surface(0, 0)
        p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
        p2_values[2]=0
        p2=Dot(p2_values, radius=0.06, fill_color=BLUE)

        g=get_grads(0,0)
        grad_viz_scale=abs(g[3]) #Maybe make arrow length proportional to gradient?
        p2_values_2=param_surface(0, 0)
        g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
                           [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
        g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
                                         axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
        g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
                                         axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

        a2=Arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

        # panel_2_shift=[-5, -3.2, 0]
        panel_2=VGroup(axes_2, curve_2, x_label_2, p2, a2) #y_label_2
        # panel_2.shift(panel_2_shift)

        # curve_2.set_stroke(opacity=0.25) 
        # self.add(panel_2)


        ## Hmm man you know what I'm pretty srue I can't just have a 3d panel mixed in these
        ## 2d ones right? It will just have to the overhead view, unless I can find some way of like 
        ## rotating it
        ## But really, the move is probably just having separate scenes that 
        ## Oh man man is there some way to actually bring these 2d plots togetehr INTO the 3d one? 
        ## Obviously dont want to add a bunch of random scope ehre, but I do think that woudl show what we're 
        ## about here really nicely. Becuase we bring together the two curves into the 3d surface and it will 
        ## be nice an dclear that the gradient arrows are guiding us downhill. 
        ## I'm also a little confused by the uphill vs downhill thing. 
        ## 
        ## Drafting a Claude prompt, I think I'm starting to figure this out - not sure what it means for 
        ## visualization and this section yet...
        ## I'm confused about why the gradient points uphill in loss landsacpes - the partial derivates 
        ## that make up the gradient are equivalent to the slope at each point - if increasing some 
        ## parameter makes the loss go down, then the slope of the loss with respect to that parameter 
        ## should be negative.
        ##
        ## Ok so it's like the slope/gradient is negative from our starting point, which means if we 
        ## add the gradient to theta, since the gradient is negative we will go uphill. 
        ## That's the deal here -> how do we visualize what's going on?
        ## I guess like adding the gradient to the parameter is not actually that natural of a thing visually?
        ## I feel like it should be though -> it should be a little vector that guides us through parameter space...
        ## Hmmmm....
        ## 
        ## Ok I think i have a better sense of what's going on here - and in the script I'll say something 
        ## like that technically the gradient goes the other way, and we'll get into this in part 2. 
        ## 
        ## Ok there's some details etc that I need to change here
        ## But I have a pretty strong vision for what I think I want the core animation to be
        ## I think doing it like this is actually going to be more clear
        ## I want the two curves to "come together" into the surfce, and then really clearly show that the 
        ## overall gradient is the combination of the cyan and yellow arrows, it will be like nicely in the
        ## center fo thes two....
        ##
        ## So.. let me build out a workign sketch that core animation, and then we can take if from there.  
        ## 

        curve_1.set_stroke(opacity=0.5)
        curve_2.set_stroke(opacity=0.5)

        panel_2.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN) #Get into 3d space so I can do the bring together animation...
        panel_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        self.add(panel_1, panel_2)

        self.frame.reorient(0, 89, 0, (-0.46, 0.0, 1.36), 8.97)

        panel_1_shift=[-5, 0, 2.0]
        panel_2_shift=[-5, 0, -2.0]
        panel_1.shift(panel_1_shift)
        panel_2.shift(panel_2_shift)

        self.wait()

        
        # self.remove(y_axis_2) #Probably fade this out during the animation. 
        
        r=panel_2.get_corner(LEFT+BOTTOM) 
        r[0]=-4.15    
        self.wait()

        self.play(panel_1.animate.shift([0, 0, -2.0]),
                  panel_2.animate.rotate(90*DEGREES, [0,0,1], about_point=r).shift([0, 0, 2.0]),
                  # panel_2.animate.shift([0, 0, 2.0]),
                  FadeOut(y_axis_2),
                  self.frame.animate.reorient(14, 79, 0, (-1.83, -0.75, 1.59), 5.01),
                  run_time=4)

        self.wait()


        panel_2.move_to(panel_1)
        panel_2.shift([0,0,-0.107]) #little nudge. 

        r=panel_2.get_corner(LEFT+BOTTOM) 
        r[0]=-4.15
        panel_2.rotate(90*DEGREES, [0,0,1], about_point=r) #If I can get rotation point right this might kidna work?


        # panel_2.rotate(-90*DEGREES, [0,0,1], about_point=r )


        reorient(132, 58, 0, (-2.77, 3.01, 2.45), 4.87)


        ## ---

        # Create main surface
        surface = ParametricSurface(
            1.5*param_surface, #Scaling here to make gradients a bit more interesting. 
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(256, 256),
        )


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
        
        # Add labels
        x_label = Tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        y_label = Tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, RIGHT)
        y_label.next_to(axes.y_axis, UP)
        z_label.next_to(axes.z_axis, OUT)
        z_label.rotate(90*DEGREES, [1,0,0])
        
        ts = TexturedSurface(surface, '/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.png')
        ts.set_shading(0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = VGroup()
        v_gridlines = VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        


        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)
        self.add(u_gridlines)
        self.add(v_gridlines)
        self.add(ts)


        self.frame.reorient(32, 59, 0, (1.88, 1.0, 1.52), 7.78)

        
        self.embed()
        self.wait(20)




# class P33v1(InteractiveScene):
#     def construct(self):

#         # Ok ok ok ok I need ot decide how automated vs manual I'm going to make this scene
#         # Thinking about this scene more, I do think the "right triangle overhead view" of the gradient is going to be nice/important.
#         #  
#         # Ok ok ok ok ok I do think i want some more Meaat on p33-p35 → specifically it’s interesting and non-obvious that we can put 
#         # each slope together in that overhead view and it will point us downhill. Worth thinking about that animation a bit I think → 
#         # maybe an actually good use case for WelchAxes lol. And i can draw a connection to part 2 → “as we’ll see in part 2 it turns 
#         # out that we can very efficiently estimate the slope of the curves without actually computing an points on them → and then 
#         # maybe the little arrows on the curves move to the overhead view? or copies of them? Them more i think about this the more 
#         # I think it should be manimy? I can draw the projections/cuts as nice blue/yellow lines in the overhead view too. 
#         #
#         # Alright kinda low on brain power here, but let me at least try to get the pieces together tonight
#         # I do think it probably makes sense to at least try to get all 3 graphs on one canvas, so I can like smoothlly move arrows arround and stuff
#         # I might need to go compute a bunch of gradients eh?


#         x_axis_1=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
#                             x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
#         y_axis_1=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
#                           y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
#         axes_1=VGroup(x_axis_1, y_axis_1)

#         points_1 = [param_surface(u, 0) for u in np.linspace(-1, 4, 128)]
#         points_mapped=np.array(points_1)[:, (0,2,1)]
#         points_mapped[:,0]=map_to_canvas(points_mapped[:,0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         points_mapped[:,1]=map_to_canvas(points_mapped[:,1], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
#         curve_1 = VMobject()
#         curve_1.set_points_smoothly(points_mapped)
#         curve_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

#         x_label_1 = Tex(r'\theta_{1}', font_size=30).set_color(CHILL_BROWN)
#         y_label_1 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
#         x_label_1.next_to(x_axis_1, RIGHT, buff=0.05)
#         y_label_1.next_to(y_axis_1, UP, buff=0.05)

#         # self.add(axes_1, curve_1)
#         # self.add(x_label_1, y_label_1)

#         # Ok, adding second curve should be pretty straightfoward, maybe I think about gradient a bit now?
#         # Intuitively I think I need to go actually compuate all the gradients at each point and cache them? 
#         # Let me go take a crack at that. 
#         # Man I guess you could show the whole gradient field with this approach - that's kinda interesting
#         # Not sure if that has a role in the video -> could be fun in the book!

#         # I've created a crappy version of c2p I think! Should be able to roll in or absorb. 
#         # Keep roling for now though, I can make it pretty after I decide if I like the pattern.

#         #Hmm well this is complicated and not working lol - will pick back up in the morning. 
#         p1_values=param_surface(0, 0)
#         p1_values[0]=map_to_canvas(p1_values[0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         p1_values[1]=map_to_canvas(p1_values[2], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)
#         p1_values[2]=0

#         p1=Dot(p1_values, radius=0.06, fill_color=YELLOW)
#         # self.add(p1)

#         g=get_grads(0,0)
#         grad_viz_scale=abs(g[2]) #Maybe make arrow length proportional to gradient?
#         p1_values_2=param_surface(0, 0)
#         g_values=np.array([[p1_values_2[0], p1_values_2[2], 0],
#                            [p1_values_2[0]+grad_viz_scale, p1_values_2[2]+grad_viz_scale*g[2]*0.6, 0]]) #Maybe I make arrow length proportional to slope or something?
#         g_values[:,0]=map_to_canvas(g_values[:,0], axis_min=x_axis_1.x_min, 
#                                          axis_max=x_axis_1.x_max, axis_end=x_axis_1.axis_length_on_canvas)
#         g_values[:,1]=map_to_canvas(g_values[:,1], axis_min=y_axis_1.y_min, 
#                                          axis_max=y_axis_1.y_max, axis_end=y_axis_1.axis_length_on_canvas)

#         a1=Arrow(start=g_values[0], end=g_values[1], fill_color=YELLOW, thickness=3.0, tip_width_ratio=5, buff=0)
#         # self.add(a1)

#         panel_1_shift=[-5, 0.4, 0]
#         panel_1=VGroup(axes_1, curve_1, x_label_1, y_label_1, p1, a1)
#         panel_1.shift(panel_1_shift)

#         curve_1.set_stroke(opacity=0.25) #Yeah so I a fade out/fade in will bascially be the optning animation for this scene?
#         self.add(panel_1)

#         # Ok I might clean up/rewrite this scene, we'll see. 
#         # I think it makes sense to to ahead and add panels 2 and 3, and then I can start thinking about how the
#         # pieces fit together, and exactly what I want to show from there
#         # This workflow feels a bit hacky/uncomfortable, but I do think I have a pretty good feeling of what I want
#         # to show, so I'm just going to trust this new/different process. 

#         x_axis_2=WelchXAxis(x_min=-1.2, x_max=4.5, x_ticks=[-1,0,1,2,3,4], x_tick_height=0.15,        
#                             x_label_font_size=24, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
#         y_axis_2=WelchYAxis(y_min=0.3, y_max=1.7, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], y_tick_width=0.15,        
#                           y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
#         axes_2=VGroup(x_axis_2, y_axis_2)


#         points_2 = [param_surface(0, v) for v in np.linspace(-1, 4, 128)]
#         points_mapped_2=np.array(points_2)[:, (1,2,0)]
#         points_mapped_2[:,0]=map_to_canvas(points_mapped_2[:,0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         points_mapped_2[:,1]=map_to_canvas(points_mapped_2[:,1], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
#         curve_2 = VMobject()
#         curve_2.set_points_smoothly(points_mapped_2)
#         curve_2.set_stroke(width=4, color=BLUE, opacity=0.8)


#         x_label_2 = Tex(r'\theta_{2}', font_size=30).set_color(CHILL_BROWN)
#         y_label_2 = Tex('Loss', font_size=25).set_color(CHILL_BROWN)
#         x_label_2.next_to(x_axis_2, RIGHT, buff=0.05)
#         y_label_2.next_to(y_axis_2, UP, buff=0.05) #not sure I need this. 

#         p2_values=param_surface(0, 0)
#         p2_values[0]=map_to_canvas(p2_values[0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         p2_values[1]=map_to_canvas(p2_values[2], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)
#         p2_values[2]=0
#         p2=Dot(p2_values, radius=0.06, fill_color=BLUE)

#         g=get_grads(0,0)
#         grad_viz_scale=abs(g[3]) #Maybe make arrow length proportional to gradient?
#         p2_values_2=param_surface(0, 0)
#         g_values_2=np.array([[p2_values_2[0], p2_values_2[2], 0],
#                            [p2_values_2[0]+grad_viz_scale, p2_values_2[2]+grad_viz_scale*g[3]*1.0, 0]]) #Maybe I make arrow length proportional to slope or something?
#         g_values_2[:,0]=map_to_canvas(g_values_2[:,0], axis_min=x_axis_2.x_min, 
#                                          axis_max=x_axis_2.x_max, axis_end=x_axis_2.axis_length_on_canvas)
#         g_values_2[:,1]=map_to_canvas(g_values_2[:,1], axis_min=y_axis_2.y_min, 
#                                          axis_max=y_axis_2.y_max, axis_end=y_axis_2.axis_length_on_canvas)

#         a2=Arrow(start=g_values_2[0], end=g_values_2[1], fill_color=BLUE, thickness=3.0, tip_width_ratio=5, buff=0)

#         panel_2_shift=[-5, -3.2, 0]
#         panel_2=VGroup(axes_2, curve_2, x_label_2, p2, a2) #y_label_2
#         panel_2.shift(panel_2_shift)

#         curve_2.set_stroke(opacity=0.25) 
#         self.add(panel_2)


#         ## Hmm man you know what I'm pretty srue I can't just have a 3d panel mixed in these
#         ## 2d ones right? It will just have to the overhead view, unless I can find some way of like 
#         ## rotating it
#         ## But really, the move is probably just having separate scenes that 
#         ## Oh man man is there some way to actually bring these 2d plots togetehr INTO the 3d one? 
#         ## Obviously dont want to add a bunch of random scope ehre, but I do think that woudl show what we're 
#         ## about here really nicely. Becuase we bring together the two curves into the 3d surface and it will 
#         ## be nice an dclear that the gradient arrows are guiding us downhill. 
#         ## I'm also a little confused by the uphill vs downhill thing. 
#         ## 
#         ## Drafting a Claude prompt, I think I'm starting to figure this out - not sure what it means for 
#         ## visualization and this section yet...
#         ## I'm confused about why the gradient points uphill in loss landsacpes - the partial derivates 
#         ## that make up the gradient are equivalent to the slope at each point - if increasing some 
#         ## parameter makes the loss go down, then the slope of the loss with respect to that parameter 
#         ## should be negative.
#         ##
#         ## Ok so it's like the slope/gradient is negative from our starting point, which means if we 
#         ## add the gradient to theta, since the gradient is negative we will go uphill. 
#         ## That's the deal here -> how do we visualize what's going on?
#         ## I guess like adding the gradient to the parameter is not actually that natural of a thing visually?
#         ## I feel like it should be though -> it should be a little vector that guides us through parameter space...
#         ## Hmmmm....
#         ## 
#         ## Ok I think i have a better sense of what's going on here - and in the script I'll say something 
#         ## like that technically the gradient goes the other way, and we'll get into this in part 2. 
#         ## 
#         ## Ok there's some details etc that I need to change here
#         ## But I have a pretty strong vision for what I think I want the core animation to be
#         ## I think doing it like this is actually going to be more clear
#         ## I want the two curves to "come together" into the surfce, and then really clearly show that the 
#         ## overall gradient is the combination of the cyan and yellow arrows, it will be like nicely in the
#         ## center fo thes two....
#         ##
#         ## So.. let me build out a workign sketch that core animation, and then we can take if from there.  
#         ## 


#         # Create main surface
#         surface = ParametricSurface(
#             1.5*param_surface, #Scaling here to make gradients a bit more interesting. 
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(256, 256),
#         )


#         # Create the surface
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 3.5, 1.0],
#             height=5,
#             width=5,
#             depth=3.5,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add labels
#         x_label = Tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
#         y_label = Tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
#         z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
#         x_label.next_to(axes.x_axis, RIGHT)
#         y_label.next_to(axes.y_axis, UP)
#         z_label.next_to(axes.z_axis, OUT)
#         z_label.rotate(90*DEGREES, [1,0,0])
        
#         ts = TexturedSurface(surface, '/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/animation/p_24_28_losses_4.png')
#         ts.set_shading(0.0, 0.1, 0)
#         ts.set_opacity(0.7)

#         # Create gridlines using polylines instead of parametric curves
#         num_lines = 20  # Number of gridlines in each direction
#         num_points = 256  # Number of points per line
#         u_gridlines = VGroup()
#         v_gridlines = VGroup()
        
#         # Create u-direction gridlines
#         u_values = np.linspace(-1, 4, num_lines)
#         v_points = np.linspace(-1, 4, num_points)
        
#         for u in u_values:
#             points = [param_surface(u, v) for v in v_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             u_gridlines.add(line)
        
#         # Create v-direction gridlines
#         u_points = np.linspace(-1, 4, num_points)
#         for v in u_values:  # Using same number of lines for both directions
#             points = [param_surface(u, v) for u in u_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             v_gridlines.add(line)

#         #i think there's a better way to do this
#         offset=surface.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
#         axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
#         # axes.move_to([1,1,1])
        


        
#         # Add everything to the scene
#         self.add(axes, x_label, y_label, z_label)
#         self.add(u_gridlines)
#         self.add(v_gridlines)
#         self.add(ts)


#         self.frame.reorient(32, 59, 0, (1.88, 1.0, 1.52), 7.78)

        
#         self.embed()
#         self.wait(20)





