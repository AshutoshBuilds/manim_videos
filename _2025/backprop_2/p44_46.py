from manimlib import *
sys.path.append('/Users/stephen/manim/videos/welch_assets')
from welch_axes import *
from functools import partial
import numpy as np
import torch

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim'
data_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/hackin'
heatmap_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim/may_27_2'

# map_min_x=0.38
# map_max_x=1.54
# map_min_y=-0.56
# map_max_y=0.56

# min_long=-7.0
# max_long=18.0
# min_lat=36.0
# max_lat=56.0


def format_number(num, total_chars=6, align='right'):
    """
    Format number to maintain consistent visual alignment for animations.
    
    Args:
        num: The number to format
        total_chars: Total character width (should accommodate largest expected number)
        align: 'right', 'left', or 'center' - how to align within the fixed width
    """
    abs_num = abs(num)
    
    # Determine appropriate precision based on magnitude
    if abs_num >= 100:
        # 100+: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: two decimal places (e.g., "0.12", "-0.12")
        formatted = f"{num:.2f}"
    
    # Pad to consistent width
    if align == 'right':
        return formatted.rjust(total_chars)
    elif align == 'left':
        return formatted.ljust(total_chars)
    else:  # center
        return formatted.center(total_chars)

def format_number_fixed_decimal(num, decimal_places=2, total_chars=6):
    """
    Alternative formatter that keeps decimal point in same position.
    Useful when you want all numbers to have the same decimal precision.
    """
    formatted = f"{num:.{decimal_places}f}"
    return formatted.rjust(total_chars)

def get_numbers(i, xs, weights, logits, yhats):
    x = xs[i, -1]
    numbers = VGroup()

    tx = Tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    numbers.add(tx)
    
    # Weights - using consistent formatting
    w = weights[i, :]
    tm1 = Tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1.scale(0.16)
    tm1.move_to([-1.195, 0.205, 0])
    numbers.add(tm1)
    
    tm2 = Tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
    tm2.scale(0.15)
    tm2.move_to([-1.155, 0.015, 0])
    numbers.add(tm2)
    
    tm3 = Tex(format_number(w[2], total_chars=6)).set_color(GREEN)
    tm3.scale(0.16)
    tm3.move_to([-1.19, -0.17, 0])
    numbers.add(tm3)
    
    # Biases
    tb1 = Tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.16)
    tb1.move_to([-0.875, 0.365, 0])
    numbers.add(tb1)
    
    tb2 = Tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.16)
    tb2.move_to([-0.875, 0.015, 0])
    numbers.add(tb2)
    
    tb3 = Tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    tb3.scale(0.16)
    tb3.move_to([-0.88, -0.335, 0])
    numbers.add(tb3)
    
    # Logits
    tl1 = Tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    numbers.add(tl1)
    
    tl2 = Tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    numbers.add(tl2)
    
    tl3 = Tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.16)  
    tl3.move_to([-0.52, -0.335, 0])
    numbers.add(tl3)
    
    # Predictions
    yhat1 = Tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = Tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = Tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    return numbers



def latlong_to_canvas(lat, long, 
                      map_min_x=0.38, map_max_x=1.54,
                      map_min_y=-0.56, map_max_y=0.56,
                      min_long=-7.0, max_long=18.0,
                      min_lat=36.0, max_lat=56.0):
    """
    Convert latitude/longitude coordinates to canvas x,y coordinates.
    
    Args:
        lat: Latitude value
        long: Longitude value
        map_min_x, map_max_x: Canvas x-coordinate bounds
        map_min_y, map_max_y: Canvas y-coordinate bounds
        min_long, max_long: Longitude bounds for the map
        min_lat, max_lat: Latitude bounds for the map
    
    Returns:
        tuple: (x, y) canvas coordinates
    """
    # Normalize longitude to [0, 1] range
    long_normalized = (long - min_long) / (max_long - min_long)
    
    # Normalize latitude to [0, 1] range
    lat_normalized = (lat - min_lat) / (max_lat - min_lat)
    
    # Map to canvas coordinates
    x = map_min_x + long_normalized * (map_max_x - map_min_x)
    y = map_min_y + lat_normalized * (map_max_y - map_min_y)
    
    return x, y


def get_grad_regions(i, ys, yhats, grads):
     #Ok let's shade some lines!
    max_region_width=0.15
    min_region_width=0.01
    region_scaling=0.15

    grad_regions = VGroup()

    y_one_hot=torch.nn.functional.one_hot(torch.tensor(int(ys[i])),3).numpy()
    dldh=yhats[i]-y_one_hot

    rh1_width=np.clip(region_scaling*np.abs(dldh[0]), min_region_width, max_region_width)
    rh1=Rectangle(0.425, rh1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rh1.move_to([-0.52, 0.37, 0])
    grad_regions.add(rh1)

    rh2_width=np.clip(region_scaling*np.abs(dldh[1]), min_region_width, max_region_width)
    rh2=Rectangle(0.425, rh2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    rh2.move_to([-0.52, 0.015, 0])
    grad_regions.add(rh2)

    rh3_width=np.clip(region_scaling*np.abs(dldh[2]), min_region_width, max_region_width)
    rh3=Rectangle(0.425, rh3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rh3.move_to([-0.52, -0.335, 0])
    grad_regions.add(rh3)

    rb1_width=np.clip(region_scaling*np.abs(grads[i,3]), min_region_width, max_region_width)
    rb1=Rectangle(0.24, rb1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rb1.move_to([-0.875, 0.37, 0])
    grad_regions.add(rb1)

    rb2_width=np.clip(region_scaling*np.abs(grads[i,4]), min_region_width, max_region_width)
    rb2=Rectangle(0.24, rb2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    rb2.move_to([-0.875, 0.015, 0])
    grad_regions.add(rb2)

    rb3_width=np.clip(region_scaling*np.abs(grads[i,5]), min_region_width, max_region_width)
    rb3=Rectangle(0.24, rb3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rb3.move_to([-0.872, -0.335, 0])
    grad_regions.add(rb3)

    rm1_width=np.clip(region_scaling*np.abs(grads[i,0]), min_region_width, max_region_width)
    rm1=Rectangle(0.42, rm1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rm1.rotate(33*DEGREES)
    rm1.move_to([-1.18, 0.20, 0])
    grad_regions.add(rm1)

    rm2_width=np.clip(region_scaling*np.abs(grads[i,1]), min_region_width, max_region_width)
    rm2=Rectangle(0.33, rm2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    # rm2.rotate(33*DEGREES)
    rm2.move_to([-1.18, 0.015, 0])
    grad_regions.add(rm2)

    rm3_width=np.clip(region_scaling*np.abs(grads[i,2]), min_region_width, max_region_width)
    rm3=Rectangle(0.42, rm3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rm3.rotate(-30.5*DEGREES)
    rm3.move_to([-1.19, -0.175, 0])
    grad_regions.add(rm3)

    return grad_regions

def get_arrow_tip(line, color=None, scale=0.1, tip_position=1.0):
    """
    Add an arrow tip to the end of a line/curve.
    
    Args:
        line: The line/curve object to add arrow to
        color: Color of the arrow tip (defaults to line's color)
        scale: Size of the arrow tip (default 0.1)
        tip_position: Position along the line (0.0 to 1.0, default 1.0 for end)
    
    Returns:
        The ArrowTip object
    """
    # Get the tip point and direction
    tip_point = line.point_from_proportion(tip_position)
    direction_point = line.point_from_proportion(tip_position - 0.05)
    direction = tip_point - direction_point
    
    # Create and position arrow tip
    arrow_tip = ArrowTip().scale(scale)
    if color is None:
        color = line.get_color()
    arrow_tip.set_color(color)
    arrow_tip.move_to(tip_point)
    arrow_tip.rotate(angle_of_vector(direction))
    
    return arrow_tip

# Create planes from line endpoints, extending only in positive y direction

def create_plane_from_line_endpoints(line, color, depth=3.0, y_extension=2.0):
    """
    Create a plane from the endpoints of an existing line object.
    The plane extends in positive y direction and in z direction (depth).
    
    Args:
        line: The existing line object
        color: Color for the plane
        depth: How far to extend in z direction (both + and -)
        y_extension: How far to extend in positive y direction
    """
    # Get the endpoints of the line
    start_point = line.get_start()
    end_point = line.get_end()
    
    # Create the four corners of our plane
    # Bottom edge (original line endpoints)
    bottom_left = start_point.copy()
    bottom_right = end_point.copy()
    
    # Top edge (extended in positive y direction)
    top_left = start_point + np.array([0, y_extension, 0])
    top_right = end_point + np.array([0, y_extension, 0])
    
    # Create a custom surface class for this rectangular plane
    class RectangularPlane(Surface):
        def __init__(self, corners, **kwargs):
            self.corners = corners
            super().__init__(
                u_range=(0, 1),
                v_range=(0, 1), 
                resolution=(20, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u goes from left to right (along the line)
            # v goes from bottom to top (y extension)
            
            # Interpolate along bottom edge
            bottom_point = interpolate(self.corners[0], self.corners[1], u)
            # Interpolate along top edge  
            top_point = interpolate(self.corners[2], self.corners[3], u)
            # Interpolate between bottom and top
            point = interpolate(bottom_point, top_point, v)
            
            return point
    
    # Create the base plane (no depth yet)
    base_plane = RectangularPlane(
        [bottom_left, bottom_right, top_left, top_right],
        color=color,
        shading=(0.2, 0.2, 0.6)
    )
    
    # Now extrude this plane in the z direction to give it depth
    class ExtrudedPlane(Surface):
        def __init__(self, base_corners, depth, **kwargs):
            self.base_corners = base_corners
            self.depth = depth
            super().__init__(
                u_range=(0, 1),
                v_range=(-1, 1),  # -1 to 1 to go from -depth/2 to +depth/2
                resolution=(20, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u parameter: interpolate along the line (left to right)
            # v parameter: interpolate in depth (z direction)
            
            # Get bottom and top points along the line at parameter u
            bottom_point = interpolate(self.base_corners[0], self.base_corners[1], u)
            top_point = interpolate(self.base_corners[2], self.base_corners[3], u)
            
            # Interpolate between bottom and top based on y-extension
            # For now, let's use the middle height
            middle_point = interpolate(bottom_point, top_point, 0.5)
            
            # Add depth in z direction
            z_offset = v * self.depth / 2
            final_point = middle_point + np.array([0, 0, z_offset])
            
            return final_point
    
    # Actually, let's make this simpler - create a plane that extends from the line
    class LineExtensionPlane(Surface):
        def __init__(self, line_start, line_end, y_extension, depth, **kwargs):
            self.line_start = line_start
            self.line_end = line_end  
            self.y_extension = y_extension
            self.depth = depth
            super().__init__(
                u_range=(0, 1),  # Along the line
                v_range=(0, 1),  # From line to extended height
                resolution=(15, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u: interpolate along original line
            line_point = interpolate(self.line_start, self.line_end, u)
            
            # v: extend in positive y direction
            extended_point = line_point + np.array([0, v * self.y_extension, 0])
            
            return extended_point
    
    return LineExtensionPlane(
        start_point, end_point, y_extension, depth,
        color=color,
        shading=(0.2, 0.2, 0.6)
    )



class p46_sketch(InteractiveScene):
    def construct(self):
        '''
        Ok starting with p45, I'll work on animating to shared p46 plot, and then start hacking on 3d. 
        '''
        data=np.load(data_path+'/cities_1d_3.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=SVGMobject(svg_path+'/p44_background_2.svg') 
        self.add(net_background)

        self.frame.reorient(0, 0, 0, (-0.03, -0.02, 0.0), 1.88)
        europe_map=ImageMobject(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)


        #Alrighty, so I think this is where it makes sense to grab welch_axes??
        # x_axis_1=WelchXAxis(x_min=-7, x_max=18, x_ticks=[], x_tick_height=0.15,        
        #                     x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        # y_axis_1=WelchYAxis(y_min=-18, y_max=10, y_ticks=[], y_tick_width=0.15,        
        #                   y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        # self.add(x_axis_1, y_axis_1)

        # Ok maybe not actually? Let me try a standard manim axis...

        axes_1 = Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )
        axes_2=Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )
        axes_3=Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_1.move_to([-0.95, 0.44, 0])
        axes_2.move_to([-0.95, 0.0, 0])
        axes_3.move_to([-0.95, -0.44, 0])
        self.add(axes_1, axes_2, axes_3)

        # for i in range(len(xs)):
        i=200 #0
            
        # if i>0:
        #     self.remove(line_1, arrow_tip_1)
        #     self.remove(line_2, arrow_tip_2)
        #     self.remove(line_3, arrow_tip_3)
        #     self.remove(nums)
        #     self.remove(heatmaps)


        nums = VGroup()
        x = xs[i, -1]
        tx = Tex(str(x) + r'^\circ')
        tx.scale(0.13)
        tx.move_to([-1.49, 0.02, 0])
        nums.add(tx)
        
        # Weights - using consistent formatting
        w = weights[i, :]
        tm1 = Tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
        tm1.scale(0.12)
        tm1.move_to([-1.185, 0.54, 0])
        nums.add(tm1)
        
        tm2 = Tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
        tm2.scale(0.12)
        tm2.move_to([-1.185, 0.1, 0])
        nums.add(tm2)
        
        tm3 = Tex(format_number(w[2], total_chars=6)).set_color(GREEN)
        tm3.scale(0.12)
        tm3.move_to([-1.185, -0.33, 0])
        nums.add(tm3)
        
        # Biases
        tb1 = Tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
        tb1.scale(0.12)
        tb1.move_to([-1.185, 0.37, 0])
        nums.add(tb1)
        
        tb2 = Tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
        tb2.scale(0.12)
        tb2.move_to([-1.185, -0.07, 0])
        nums.add(tb2)
        
        tb3 = Tex(format_number(w[5], total_chars=6)).set_color(GREEN)
        tb3.scale(0.12)
        tb3.move_to([-1.185, -0.51, 0])
        nums.add(tb3)
        
        # Logits
        tl1 = Tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
        tl1.scale(0.16)
        tl1.move_to([-0.52, 0.37, 0])
        nums.add(tl1)
        
        tl2 = Tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
        tl2.scale(0.16)
        tl2.move_to([-0.52, 0.015, 0])
        nums.add(tl2)
        
        tl3 = Tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
        tl3.scale(0.16)  
        tl3.move_to([-0.52, -0.335, 0])
        nums.add(tl3)
        
        # Predictions
        yhat1 = Tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
        yhat1.scale(0.16)
        yhat1.move_to([0.22, 0.37, 0])
        nums.add(yhat1)
        
        yhat2 = Tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
        yhat2.scale(0.16)
        yhat2.move_to([0.22, 0.015, 0])
        nums.add(yhat2)
        
        yhat3 = Tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
        yhat3.scale(0.16)
        yhat3.move_to([0.22, -0.335, 0])
        nums.add(yhat3)


    
        def line_function_1(x): return weights[i,0] * x + weights[i,3]
        line_1 = axes_1.get_graph(line_function_1, stroke_width=3, color='#00FFFF', x_range=[-14, 14])
        arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.08)

        def line_function_2(x): return weights[i,1] * x + weights[i,4]
        line_2 = axes_2.get_graph(line_function_2, stroke_width=3, color=YELLOW, x_range=[-14, 14])
        arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.08)

        def line_function_3(x): return weights[i,2] * x + weights[i,5]
        line_3 = axes_3.get_graph(line_function_3, stroke_width=3, color=GREEN, x_range=[-14, 14])
        arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.08)

        heatmaps=Group()
        heatmap_yhat3=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_3.png')
        heatmap_yhat3.scale([0.29, 0.28, 0.28])
        heatmap_yhat3.move_to([0.96,0,0])
        heatmap_yhat3.set_opacity(0.5)
        heatmaps.add(heatmap_yhat3)

        heatmap_yhat1=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_1.png')
        heatmap_yhat1.scale([0.29, 0.28, 0.28])
        heatmap_yhat1.move_to([0.96,0,0])
        heatmap_yhat1.set_opacity(0.5)
        heatmaps.add(heatmap_yhat1)

        heatmap_yhat2=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_2.png')
        heatmap_yhat2.scale([0.29, 0.28, 0.28])
        heatmap_yhat2.move_to([0.96,0,0])
        heatmap_yhat2.set_opacity(0.5)
        heatmaps.add(heatmap_yhat2)


        self.add(axes_1, line_1, arrow_tip_1)
        self.add(axes_2, line_2, arrow_tip_2)
        self.add(axes_3, line_3, arrow_tip_3)
        self.add(nums)
        self.add(heatmaps)
        self.wait(0.1)


        # Ok start working through this animation 
        # Drop bounding boxes, bring my plots together into one, 
        # I guess drop 2 axes - i also want to zoom in and I think we'll 
        # add some ticks etc -> either in pure manim or AE exports. 

        top_plot_group=VGroup(axes_1, line_1, arrow_tip_1)
        middle_plot_group=VGroup(axes_2, line_2, arrow_tip_2)
        bottom_plot_group=VGroup(axes_3, line_3, arrow_tip_3)
        

        background_elements_to_remove=background_elements_to_remove = [31, 32, 33, 34, 35, 36, 45, 46, 47, 48, 49, 50, 82, 83, 84, 85, 86, 89, 90]
        background_elements_to_keep = [i for i in range(len(net_background)) if i not in background_elements_to_remove]
        self.wait()

        self.play(top_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  middle_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  bottom_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  *[FadeOut(net_background[o]) for o in background_elements_to_remove],
                  *[net_background[o].animate.set_opacity(0.5) for o in background_elements_to_keep],
                  FadeOut(nums[1:7]),
                  nums[0].animate.set_opacity(0.3), 
                  nums[7:].animate.set_opacity(0.3),
                  # arrow_tip_1.animate.scale(0.7),
                  # arrow_tip_2.animate.scale(0.7),
                  # arrow_tip_3.animate.scale(0.7),
                  self.frame.animate.reorient(0, 0, 0, (-0.64, 0.0, 0.0), 1.14),
                  run_time=4.0
            )

        self.remove(axes_1, axes_3)
        self.remove(arrow_tip_2) #Occlusions
        self.add(arrow_tip_2)
        self.wait()

        # Originally i was thinking it would clarify things if i turned down the opacity on the lines, I think it actually makes thing more confusion though.
        # Not sure wheniff I'll be able to scale arrow heads now -> one option would be to just start with them at a lower opacity!
        # self.play(line_1.animate.set_opacity(0.5),
        #           line_2.animate.set_opacity(0.5),
        #           line_3.animate.set_opacity(0.5),
        #           arrow_tip_1.animate.scale(0.7).set_opacity(0.5),
        #           arrow_tip_2.animate.scale(0.7).set_opacity(0.5),
        #           arrow_tip_3.animate.scale(0.7).set_opacity(0.5),
        #           run_time=1.0
        #       )
        # self.wait()


        #Ok so now we ad in overlays showing mapping
        overlays_1=SVGMobject(svg_path+'/p46_overlays_1.svg') 
        overlays_2=SVGMobject(svg_path+'/p46_overlays_2.svg') 

        overlays_1.scale(0.57)
        overlays_1.move_to([-0.64, 0.003, 0])
        self.add(overlays_1[1:])
        #Ah would be cool to bring back up th opacity on h1 and y hat 1 when i mention them in the script!
        nums[0].set_opacity(1.0) #oes it still make sense to turn this down in the first place?
        nums[7].set_opacity(1.0) #Will need to fiddle with timing a bit, turning up opacity on 2.34
        self.wait()


        #Then run training in this view, with city labels on axis. 

        self.play(line_1.animate.set_opacity(1.0),
                  line_2.animate.set_opacity(1.0),
                  line_3.animate.set_opacity(1.0),
                  arrow_tip_1.animate.set_opacity(1.0),
                  arrow_tip_2.animate.set_opacity(1.0),
                  arrow_tip_3.animate.set_opacity(1.0),
                  run_time=1.0
              )
        self.remove(arrow_tip_2) #Occlusions
        self.add(arrow_tip_2)
        self.wait()

        # So i think for the training run I want to zoom in even further, and maybe everything fades to black? 
        # Put these together into a single a move, and consider running trianing while camera moves
        # That will either be overwhelming or cool, definitely worht a try!
        self.frame.reorient(0, 0, 0, (-1.02, 0.02, 0.0), 0.62)
        self.play(*[net_background[o].animate.set_opacity(0.0) for o in background_elements_to_keep])
        self.play(nums[0].animate.set_opacity(0.0))
        self.play(nums[7:].animate.set_opacity(0.0))
        self.play(overlays_1[1:].animate.set_opacity(0.0)) #It might be nice to bring along the +/-10s with the camera move, we'll see. 

        #Now add new overlays as we land the camera move and fade outs. 
        overlays_2.scale(0.30)
        overlays_2.move_to([-1.02, 0.025, 0])
        self.add(overlays_2[1:])

        self.wait()

        # Ok this looks pretty good, major components of the sketch are coming together. 
        # Now, the most complex transition, maybe in the whole video -> but it's going to be dope,
        # is going from this 2d view to a 3d planes over the map view, and then morphing these guys into softmax curvy planes. 
        # It's a quick line in the script (might exapnd, we'll see) - but I think it's pretty important pedagogically 
        # I want to make it really visceral that we're just fitting planes, because this as about to break when we go to Barlay Hertog
        # Now, how the FUCK an I going to turn my lines into planes, my a-axis into the map, and now that I'm looking at it, I really think the cool move here
        # is to have the little points and ideally the labels for each city move to their locations on the map!
        # Do I might have to play that whole game again that I did last time were where all the 2d stuff has actually been rotated up
        # Unclear to me at point I need to switch to this viewpoint -> might be all the way through -> well see. 
        # Let me start though by rotating what I have. 

        stuff_to_rotate=VGroup(axes_2, line_1, line_2, line_3, arrow_tip_1, arrow_tip_2, arrow_tip_3)
        stuff_to_rotate.rotate(90*DEGREES, [1, 0, 0])
        overlays_2.rotate(90*DEGREES, [1, 0, 0])
        self.frame.reorient(0, 90, 0, (-1.0, 0.02, -0.0), 0.62)
        self.wait()

        # Ok so the move I'm kinda seeing in my had is camera pans to the left, and the planes and map "grow/exapnd out of the back of the lines"
        # The map could also just be a reveal, right? and then points move over?
        # The reveal is simple, let me try that first. 

        self.remove(heatmaps)
        # self.remove(europe_map) #I could move over the one i already have, but that seems like more complexity than I need. Well shit it makes ordering weird if I reimport actually? 
        # europe_map_2=ImageMobject(svg_path +'/map_cropped_one.png')
        # europe_map_2.scale(0.1)
        # europe_map_2.move_to([-1, 0, 0])
        # self.remove(stuff_to_rotate, overlays_2)
        # self.add(europe_map_2)
        # self.add(stuff_to_rotate, overlays_2)

        europe_map.scale(0.4)
        europe_map.move_to([-1, 0.23, 0]) #Not going to fine tune too much here until I have the final map. 

        self.frame.reorient(-30, 59, 0, (-1.02, 0.04, -0.0), 0.62) #So this can be a cool pan to the side/reveal. 

        ## Hmm make planes or dots/maybe labels next? 
        ## Let's take a rough crack at dots/labels. 
        ## Most obvous thing to do here is to just move my 2d labels, this might look fine 
        ## I'm tempted to change the grouping/layering first though in illustrator instead of sifting through all the elements manually. 
        ## Yeah let me do that next
        # self.remove(overlays_2[:30])

        # self.remove(overlays_2[1:18]) #Madrid label
        # self.remove(overlays_2[18:33] #Paris label
        # self.remove(overlays_2[33:50]) #Berlin label
        madrid_label=overlays_2[1:18]
        madrid_center=madrid_label.get_center()
        madrid_label.move_to([-1.07489443,  0.1     , -0.06620278])

        paris_label=overlays_2[18:33]
        paris_label.move_to([-0.98353332,  0.3     , -0.06596944])

        berlin_label=overlays_2[33:50]
        berlin_label.move_to([-0.83665553,  0.4     , -0.06464722])

        # Ok i can animated those moves later depending on how stuff shakes out. 
        # Now, how do I extend my lintes to be planes?
        # Create planes from your existing lines
        # plane_1 = create_plane_from_line_endpoints(line_1, '#00FFFF', depth=2.0, y_extension=0.5)
        # plane_2 = create_plane_from_line_endpoints(line_2, YELLOW, depth=2.0, y_extension=0.5)
        # plane_3 = create_plane_from_line_endpoints(line_3, GREEN, depth=2.0, y_extension=0.5)
        # plane_1.set_opacity(0.3)
        # plane_2.set_opacity(0.3)
        # plane_3.set_opacity(0.3)

        # self.add(plane_1, plane_2, plane_3)

        #Ok dope, basic planes are working -> my gut here is that we'll want to actually lose the city labels - I'll test when I come through on the next pass. 
        #Now I do think animating these planes growing "out of the lines" is pretty helpful/important - let me take a crack at that now. 
        # Can i just like scale the plane around the arrow axis and then have to animate/scale out?
        self.frame.reorient(11, 60, 0, (-0.98, 0.06, 0.02), 0.62)


        plane_1_zero = create_plane_from_line_endpoints(line_1, '#00FFFF', y_extension=0.01)
        plane_1_full = create_plane_from_line_endpoints(line_1, '#00FFFF', y_extension=0.5)
        plane_1_zero.set_opacity(0.3)
        plane_1_full.set_opacity(0.3)

        self.wait()
        self.play(Transform(plane_1_zero, plane_1_full), run_time=2)
        self.wait()

        #Nice that works! I can noodle a bit, but I think i like the idea of them coming in sequentially. Might work better with some script tweaks - no big deal. 


        plane_2_zero = create_plane_from_line_endpoints(line_2, YELLOW, y_extension=0.01)
        plane_2_full = create_plane_from_line_endpoints(line_2, YELLOW, y_extension=0.5)
        plane_2_zero.set_opacity(0.3)
        plane_2_full.set_opacity(0.3)

        self.wait()
        self.play(Transform(plane_2_zero, plane_2_full), run_time=2)
        self.wait()

        plane_3_zero = create_plane_from_line_endpoints(line_3, GREEN, y_extension=0.01)
        plane_3_full = create_plane_from_line_endpoints(line_3, GREEN, y_extension=0.5)
        plane_3_zero.set_opacity(0.3)
        plane_3_full.set_opacity(0.3)

        self.wait()
        self.play(Transform(plane_3_zero, plane_3_full), run_time=2)
        self.wait()










        # self.play(top_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
        #           middle_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
        #           bottom_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
        #           # FadeOut(axes_1),
        #           # FadeOut(axes_3),
        #           *[FadeOut(net_background[o]) for o in background_elements_to_remove],
        #           *[net_background[o].animate.set_opacity(0.5) for o in background_elements_to_keep],
        #           # FadeOut(net_background[84:87]),
        #           # FadeOut(net_background[89:91]),
        #           # FadeOut(net_background[83]),
        #           # FadeOut(net_background[82]),
        #           # FadeOut(net_background[32:37]),
        #           # FadeOut(net_background[45:51]),
        #           # FadeOut(net_background[31]),
        #           FadeOut(nums[1:7]),
        #           # top_plot_group.animate.scale(1.5),
        #           # middle_plot_group.animate.scale(1.5),
        #           # bottom_plot_group.animate.scale(1.5),
        #           # arrow_tip_1.animate.scale(0.7),
        #           # arrow_tip_2.animate.scale(0.7),
        #           # arrow_tip_3.animate.scale(0.7),
        #           self.frame.animate.reorient(0, 0, 0, (-0.64, 0.0, 0.0), 1.14),
        #           run_time=4.0
        #     )

        # self.play(top_plot_group.animate.move_to([-0.95, 0.0, 0]), run_time=3.0)
        # self.play(bottom_plot_group.animate.move_to([-0.95, 0.0, 0]), run_time=3.0)

        # self.play(FadeOut(axes_1))
        # self.play(FadeOut(axes_3))

        # self.wait()

        # self.remove(net_background[84:87]) #Bottom surrounding square, some arrows
        # self.remove(net_background[89:91])
        # self.remove(net_background[83])
        # self.remove(net_background[82])

        # self.remove(net_background[32:37])
        # self.remove(net_background[45:51])
        # self.remove(net_background[31]) #Random m

        # self.remove(nums[1:7])

        # top_plot_group.scale(1.5)
        # middle_plot_group.scale(1.5)
        # bottom_plot_group.scale(1.5)
        # arrow_tip_1.scale(0.7)
        # arrow_tip_2.scale(0.7)
        # arrow_tip_3.scale(0.7)

        # middle_plot_group.move_to([-1.0, 0.025, 0])
        # top_plot_group.move_to([-1.0, 0.025, 0])
        # bottom_plot_group.move_to([-1.0, 0.025, 0])

        # self.frame.reorient(0, 0, 0, (-0.64, 0.0, 0.0), 1.14)

        #Ok yeah so I think this is going to be a progressive zoom in deal. So i think we'll have our first layer of zoom, then 
        # I'll add some labels and talk about the mapping from x to input to output, and add some labels etc, then 
        # I'll train and zoom in - maybe at the same time we'll see. 
        # Then end with nice clear labels for the 3 regions below 
        # Want to make it really clear it's longitude!






        self.wait()
        self.embed()


class p45_sketch(InteractiveScene):
    def construct(self):
        '''
        Ok so here i want to get the basic elements of the little linear models working
        Then I can working on smoothly transition from ball and stick to linaer -> shouldn't be terrible
        Before I do that I can work on some writing and decide if want to add any real time curves
        Finally, the most difficult/important thing is what I build on this next - 
        Brining 3 lines together, longitudes on bottom
        Then exapnd to planes -> that should be interesting
        Finally put over map, and morph smoothly to sofmax version - that should be interesting. 
        '''
        data=np.load(data_path+'/cities_1d_3.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=SVGMobject(svg_path+'/p44_background_2.svg') 
        self.add(net_background)

        self.frame.reorient(0, 0, 0, (-0.03, -0.02, 0.0), 1.88)
        europe_map=ImageMobject(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)


        #Alrighty, so I think this is where it makes sense to grab welch_axes??
        # x_axis_1=WelchXAxis(x_min=-7, x_max=18, x_ticks=[], x_tick_height=0.15,        
        #                     x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=5)
        # y_axis_1=WelchYAxis(y_min=-18, y_max=10, y_ticks=[], y_tick_width=0.15,        
        #                   y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)
        # self.add(x_axis_1, y_axis_1)

        # Ok maybe not actually? Let me try a standard manim axis...

        axes_1 = Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_2=Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_3=Axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )

        axes_1.move_to([-0.95, 0.44, 0])
        axes_2.move_to([-0.95, 0.0, 0])
        axes_3.move_to([-0.95, -0.44, 0])
        self.add(axes_1, axes_2, axes_3)

        for i in range(len(xs)):
            if i>0:
                self.remove(line_1, arrow_tip_1)
                self.remove(line_2, arrow_tip_2)
                self.remove(line_3, arrow_tip_3)
                self.remove(nums)
                self.remove(heatmaps)

  
            nums = VGroup()
            x = xs[i, -1]
            tx = Tex(str(x) + r'^\circ')
            tx.scale(0.13)
            tx.move_to([-1.49, 0.02, 0])
            nums.add(tx)
            
            # Weights - using consistent formatting
            w = weights[i, :]
            tm1 = Tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
            tm1.scale(0.12)
            tm1.move_to([-1.185, 0.54, 0])
            nums.add(tm1)
            
            tm2 = Tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
            tm2.scale(0.12)
            tm2.move_to([-1.185, 0.1, 0])
            nums.add(tm2)
            
            tm3 = Tex(format_number(w[2], total_chars=6)).set_color(GREEN)
            tm3.scale(0.12)
            tm3.move_to([-1.185, -0.33, 0])
            nums.add(tm3)
            
            # Biases
            tb1 = Tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
            tb1.scale(0.12)
            tb1.move_to([-1.185, 0.37, 0])
            nums.add(tb1)
            
            tb2 = Tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
            tb2.scale(0.12)
            tb2.move_to([-1.185, -0.07, 0])
            nums.add(tb2)
            
            tb3 = Tex(format_number(w[5], total_chars=6)).set_color(GREEN)
            tb3.scale(0.12)
            tb3.move_to([-1.185, -0.51, 0])
            nums.add(tb3)
            
            # Logits
            tl1 = Tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
            tl1.scale(0.16)
            tl1.move_to([-0.52, 0.37, 0])
            nums.add(tl1)
            
            tl2 = Tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
            tl2.scale(0.16)
            tl2.move_to([-0.52, 0.015, 0])
            nums.add(tl2)
            
            tl3 = Tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
            tl3.scale(0.16)  
            tl3.move_to([-0.52, -0.335, 0])
            nums.add(tl3)
            
            # Predictions
            yhat1 = Tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
            yhat1.scale(0.16)
            yhat1.move_to([0.22, 0.37, 0])
            nums.add(yhat1)
            
            yhat2 = Tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
            yhat2.scale(0.16)
            yhat2.move_to([0.22, 0.015, 0])
            nums.add(yhat2)
            
            yhat3 = Tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
            yhat3.scale(0.16)
            yhat3.move_to([0.22, -0.335, 0])
            nums.add(yhat3)


        
            def line_function_1(x): return weights[i,0] * x + weights[i,3]
            line_1 = axes_1.get_graph(line_function_1, color='#00FFFF', x_range=[-12, 12])
            arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.1)

            def line_function_2(x): return weights[i,1] * x + weights[i,4]
            line_2 = axes_2.get_graph(line_function_2, color=YELLOW, x_range=[-12, 12])
            arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.1)

            def line_function_3(x): return weights[i,2] * x + weights[i,5]
            line_3 = axes_3.get_graph(line_function_3, color=GREEN, x_range=[-12, 12])
            arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.1)

            heatmaps=Group()
            heatmap_yhat3=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_3.png')
            heatmap_yhat3.scale([0.29, 0.28, 0.28])
            heatmap_yhat3.move_to([0.96,0,0])
            heatmap_yhat3.set_opacity(0.5)
            heatmaps.add(heatmap_yhat3)

            heatmap_yhat1=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_1.png')
            heatmap_yhat1.scale([0.29, 0.28, 0.28])
            heatmap_yhat1.move_to([0.96,0,0])
            heatmap_yhat1.set_opacity(0.5)
            heatmaps.add(heatmap_yhat1)

            heatmap_yhat2=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_2.png')
            heatmap_yhat2.scale([0.29, 0.28, 0.28])
            heatmap_yhat2.move_to([0.96,0,0])
            heatmap_yhat2.set_opacity(0.5)
            heatmaps.add(heatmap_yhat2)


            self.add(axes_1, line_1, arrow_tip_1)
            self.add(axes_2, line_2, arrow_tip_2)
            self.add(axes_3, line_3, arrow_tip_3)
            self.add(nums)
            self.add(heatmaps)
            self.wait(0.1)




        self.wait()
        self.embed()


class p44_v2(InteractiveScene):
    def construct(self):

        data=np.load(data_path+'/cities_1d_2.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.frame.reorient(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        self.frame.reorient(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)
    
        self.frame.reorient(0, 0, 0, (-0.03, -0.02, 0.0), 1.88)
        europe_map=ImageMobject(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)


        for i in range(len(xs)):
            if i>0:
                self.remove(nums)
                self.remove(grad_regions)
                self.remove(heatmaps)
                self.remove(training_point) 
                self.remove(step_label,step_count)  
                # self.wait(0.1)       

            nums=get_numbers(i, xs, weights, logits, yhats)
            grad_regions=get_grad_regions(i, ys, yhats, grads)
            
            #Ok how do I do a cool surface again? Should i actually just try images first real quick?
            #Ok maybe images are fine for a bit - I'll switch to real surfaces when I need to

            heatmaps=Group()
            heatmap_yhat3=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_3.png')
            heatmap_yhat3.scale([0.29, 0.28, 0.28])
            heatmap_yhat3.move_to([0.96,0,0])
            heatmap_yhat3.set_opacity(0.5)
            heatmaps.add(heatmap_yhat3)

            heatmap_yhat1=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_1.png')
            heatmap_yhat1.scale([0.29, 0.28, 0.28])
            heatmap_yhat1.move_to([0.96,0,0])
            heatmap_yhat1.set_opacity(0.5)
            heatmaps.add(heatmap_yhat1)

            heatmap_yhat2=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_2.png')
            heatmap_yhat2.scale([0.29, 0.28, 0.28])
            heatmap_yhat2.move_to([0.96,0,0])
            heatmap_yhat2.set_opacity(0.5)
            heatmaps.add(heatmap_yhat2)

            #Ok, last piece of the puzzle here for a basic demo is to scatter the training points - let's go!
            #Ok this is kinda hacky/dumb - but let's just do little manual calibration to find corers on canvas
            #Then it's just an affine transform from boundaries - right?
            # bottom_left=Dot([0.38, -0.56, 0], radius=0.007)
            # self.add(bottom_left)
            # top_left=Dot([0.38, 0.56, 0], radius=0.007)
            # self.add(top_left)
            # bottom_right=Dot([1.54, -0.56, 0], radius=0.007)
            # self.add(bottom_right)
            # top_right=Dot([1.54, 0.56, 0], radius=0.007)
            # self.add(top_right)

            canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1])
            training_point=Dot([canvas_x, canvas_y, 0], radius=0.012)
            if ys[i]==0.0: training_point.set_color('#00FFFF')
            elif ys[i]==1.0: training_point.set_color(YELLOW)
            elif ys[i]==2.0: training_point.set_color(GREEN)   

            step_label=Text("Step=")  
            step_label.set_color(CHILL_BROWN)
            step_label.scale(0.12)
            step_label.move_to([1.3, -0.85, 0])

            step_count=Text(str(i).zfill(3))
            step_count.set_color(CHILL_BROWN)
            step_count.scale(0.12)
            step_count.move_to([1.43, -0.85, 0])

            self.add(step_label,step_count) 


            self.add(nums)
            self.add(grad_regions)
            self.add(heatmaps)
            self.add(training_point)
            self.wait(0.1)


        self.wait()
        self.embed()




class p44_v1(InteractiveScene):
    def construct(self):

        data=np.load(data_path+'/cities_1d_1.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.frame.reorient(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        self.frame.reorient(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)

        #Ok, let's render some numbers - probably wrap up some of this rendering into a few methods -eh?
        i=4
        x=xs[i, -1]
        tx = Tex(str(x) + r'^\circ')
        tx.scale(0.13)
        tx.move_to([-1.49, 0.02, 0])
        self.add(tx)

        w=weights[i,:]
        tm1=Tex(format_number(w[0])).set_color('#00FFFF')
        tm1.scale(0.16)
        tm1.move_to([-1.195, 0.205, 0])
        self.add(tm1)

        tm2=Tex(format_number(w[1])).set_color(YELLOW)
        tm2.scale(0.15)
        tm2.move_to([-1.155, 0.015, 0])
        self.add(tm2)

        tm3=Tex(format_number(w[2])).set_color(GREEN)
        tm3.scale(0.16)
        tm3.move_to([-1.19, -0.17, 0])
        self.add(tm3)

        tb1=Tex(format_number(w[3])).set_color('#00FFFF')
        tb1.scale(0.16)
        tb1.move_to([-0.875, 0.365, 0])
        self.add(tb1)

        tb2=Tex(format_number(w[4])).set_color(YELLOW)
        tb2.scale(0.16)
        tb2.move_to([-0.875, 0.015, 0])
        self.add(tb2)

        tb3=Tex(format_number(w[5])).set_color(GREEN)
        tb3.scale(0.16)
        tb3.move_to([-0.88, -0.335, 0])
        self.add(tb3)

        tl1=Tex(format_number(logits[i,0])).set_color('#00FFFF')
        tl1.scale(0.16)
        tl1.move_to([-0.52, 0.37, 0])
        self.add(tl1)

        tl2=Tex(format_number(logits[i,1])).set_color(YELLOW)
        tl2.scale(0.16)
        tl2.move_to([-0.52, 0.015, 0])
        self.add(tl2)

        tl3=Tex(format_number(logits[i,2])).set_color(GREEN)
        tl3.scale(0.16)
        tl3.move_to([-0.52, -0.335, 0])
        self.add(tl3)

        yhat1=Tex(format_number(yhats[i,0])).set_color('#00FFFF')
        yhat1.scale(0.16)
        yhat1.move_to([0.22, 0.37, 0])
        self.add(yhat1)

        yhat2=Tex(format_number(yhats[i,1])).set_color(YELLOW)
        yhat2.scale(0.16)
        yhat2.move_to([0.22, 0.015, 0])
        self.add(yhat2)

        yhat3=Tex(format_number(yhats[i,2])).set_color(GREEN)
        yhat3.scale(0.16)
        yhat3.move_to([0.22, -0.335, 0])
        self.add(yhat3)

        #Ok let's shade some lines!
        max_region_width=0.15
        min_region_width=0.01
        region_scaling=0.15

        y_one_hot=torch.nn.functional.one_hot(torch.tensor(int(ys[i])),3).numpy()
        dldh=yhats[i]-y_one_hot

        rh1_width=np.clip(region_scaling*np.abs(dldh[0]), min_region_width, max_region_width)
        rh1=Rectangle(0.425, rh1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
        rh1.move_to([-0.52, 0.37, 0])
        self.add(rh1)

        rh2_width=np.clip(region_scaling*np.abs(dldh[1]), min_region_width, max_region_width)
        rh2=Rectangle(0.425, rh2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
        rh2.move_to([-0.52, 0.015, 0])
        self.add(rh2)

        rh3_width=np.clip(region_scaling*np.abs(dldh[2]), min_region_width, max_region_width)
        rh3=Rectangle(0.425, rh3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
        rh3.move_to([-0.52, -0.335, 0])
        self.add(rh3)

        rb1_width=np.clip(region_scaling*np.abs(grads[i,3]), min_region_width, max_region_width)
        rb1=Rectangle(0.24, rb1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
        rb1.move_to([-0.875, 0.37, 0])
        self.add(rb1)

        rb2_width=np.clip(region_scaling*np.abs(grads[i,4]), min_region_width, max_region_width)
        rb2=Rectangle(0.24, rb2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
        rb2.move_to([-0.875, 0.015, 0])
        self.add(rb2)

        rb3_width=np.clip(region_scaling*np.abs(grads[i,5]), min_region_width, max_region_width)
        rb3=Rectangle(0.24, rb3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
        rb3.move_to([-0.872, -0.335, 0])
        self.add(rb3)

        rm1_width=np.clip(region_scaling*np.abs(grads[i,0]), min_region_width, max_region_width)
        rm1=Rectangle(0.42, rm1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
        rm1.rotate(33*DEGREES)
        rm1.move_to([-1.18, 0.20, 0])
        self.add(rm1)

        rm2_width=np.clip(region_scaling*np.abs(grads[i,1]), min_region_width, max_region_width)
        rm2=Rectangle(0.33, rm2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
        # rm2.rotate(33*DEGREES)
        rm2.move_to([-1.18, 0.015, 0])
        self.add(rm2)

        rm3_width=np.clip(region_scaling*np.abs(grads[i,2]), min_region_width, max_region_width)
        rm3=Rectangle(0.42, rm3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
        rm3.rotate(-30.5*DEGREES)
        rm3.move_to([-1.19, -0.175, 0])
        self.add(rm3)

        ## Hmm i probaby a little gradient width demo kinda deal? yeah i think that makes sense
        ## Let me try to get a full-ish training demo togther first though to flush out any big issues. 
        ## There's a bunch of different ways I could color the points -> kinda leaning towards coloring 
        ## them according to their labels
        ## For rendering the heatmaps, I know that I'm going to want them to be 3d in a few paragraphs, 
        ## so I'm tempted to just have them b3 3d the whole time, just width all zeros for the z value to 
        ## start out. 

        self.frame.reorient(0, 0, 0, (-0.03, -0.02, 0.0), 1.88)
        europe_map=ImageMobject(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])

        self.add(europe_map)

        #Ok how do I do a cool surface again? Should i actually just try images first real quick?
        #Ok maybe images are fine for a bit - I'll switch to real surfaces when I need to

        heatmap_yhat3=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_3.png')
        heatmap_yhat3.scale([0.29, 0.28, 0.28])
        heatmap_yhat3.move_to([0.96,0,0])
        heatmap_yhat3.set_opacity(0.5)
        self.add(heatmap_yhat3)

        heatmap_yhat1=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_1.png')
        heatmap_yhat1.scale([0.29, 0.28, 0.28])
        heatmap_yhat1.move_to([0.96,0,0])
        heatmap_yhat1.set_opacity(0.5)
        self.add(heatmap_yhat1)

        heatmap_yhat2=ImageMobject(heatmap_path +'/'+str(i)+'_yhat_2.png')
        heatmap_yhat2.scale([0.29, 0.28, 0.28])
        heatmap_yhat2.move_to([0.96,0,0])
        heatmap_yhat2.set_opacity(0.5)
        self.add(heatmap_yhat2)

        #Ok, last piece of the puzzle here for a basic demo is to scatter the training points - let's go!
        #Ok this is kinda hacky/dumb - but let's just do little manual calibration to find corers on canvas
        #Then it's just an affine transform from boundaries - right?
        # bottom_left=Dot([0.38, -0.56, 0], radius=0.007)
        # self.add(bottom_left)
        # top_left=Dot([0.38, 0.56, 0], radius=0.007)
        # self.add(top_left)
        # bottom_right=Dot([1.54, -0.56, 0], radius=0.007)
        # self.add(bottom_right)
        # top_right=Dot([1.54, 0.56, 0], radius=0.007)
        # self.add(top_right)


        canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1])
        training_point=Dot([canvas_x, canvas_y, 0], radius=0.012)
        if ys[i]==0.0: training_point.set_color('#00FFFF')
        elif ys[i]==1.0: training_point.set_color(YELLOW)
        elif ys[i]==2.0: training_point.set_color(GREEN)
        self.add(training_point)





        self.wait()
        self.embed()