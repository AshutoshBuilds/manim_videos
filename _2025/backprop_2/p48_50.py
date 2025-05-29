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
heatmap_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim/may_28_1'


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


class p46_sketch(InteractiveScene):
    def construct(self):
        '''
        Ok starting with p45, I'll work on animating to shared p46 plot, and then start hacking on 3d. 
        '''
        data=np.load(data_path+'/cities_2d_1.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:15]
        grads=data[:,15:27]
        logits=data[:,27:31]
        yhats=data[:, 31:]


        net_background=SVGMobject(svg_path+'/p_48_background_1.svg') 
        self.add(net_background)

        self.frame.reorient(0, 0, 0, (-0.03, 0.01, 0.0), 2.01)
        europe_map=ImageMobject(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)


        i=0

        x1 = xs[i, 0]  # First input feature
        x2 = xs[i, 1]  # Second input feature
        
        nums = VGroup()
        
        # First input value
        tx1 = Tex(str(x1) + r'^\circ')
        tx1.scale(0.12)
        tx1.move_to([-1.53, 0.155, 0])  
        nums.add(tx1)
        
        # Second input value
        tx2 = Tex(str(x2) + r'^\circ')
        tx2.scale(0.12)
        tx2.move_to([-1.52, -0.19, 0])  
        nums.add(tx2)

        #   Neuron 1 weights (cyan)
        w = weights[i, :]
        tm1_1 = Tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
        tm1_1.scale(0.12)
        tm1_1.move_to([-1.04, 0.85, 0])
        nums.add(tm1_1)
        
        tm1_2 = Tex(format_number(w[1], total_chars=6)).set_color('#00FFFF')
        tm1_2.scale(0.12)
        tm1_2.move_to([-1.04, 0.72, 0])
        nums.add(tm1_2)       

        tb1 = Tex(format_number(w[8], total_chars=6)).set_color('#00FFFF')
        tb1.scale(0.12)
        tb1.move_to([-1.04, 0.59, 0])
        nums.add(tb1)

        tm2_1 = Tex(format_number(w[2], total_chars=6)).set_color(YELLOW)
        tm2_1.scale(0.12)
        tm2_1.move_to([-1.04, 0.38, 0])
        nums.add(tm2_1)
        
        tm2_2 = Tex(format_number(w[3], total_chars=6)).set_color(YELLOW)
        tm2_2.scale(0.12)
        tm2_2.move_to([-1.04, 0.25, 0])
        nums.add(tm2_2)       

        tb2 = Tex(format_number(w[9], total_chars=6)).set_color(YELLOW)
        tb2.scale(0.12)
        tb2.move_to([-1.04, 0.12, 0])
        nums.add(tb2)

        tm3_1 = Tex(format_number(w[4], total_chars=6)).set_color(GREEN)
        tm3_1.scale(0.12)
        tm3_1.move_to([-1.04, -0.08, 0])
        nums.add(tm3_1)
        
        t3_2 = Tex(format_number(w[5], total_chars=6)).set_color(GREEN)
        t3_2.scale(0.12)
        t3_2.move_to([-1.04, -0.21, 0])
        nums.add(t3_2)       

        tb3 = Tex(format_number(w[10], total_chars=6)).set_color(GREEN)
        tb3.scale(0.12)
        tb3.move_to([-1.04, -0.34, 0])
        nums.add(tb3)

        tm4_1 = Tex(format_number(w[6], total_chars=6)).set_color('#FF00FF')
        tm4_1.scale(0.12)
        tm4_1.move_to([-1.04, -0.54, 0])
        nums.add(tm4_1)
        
        t4_2 = Tex(format_number(w[7], total_chars=6)).set_color('#FF00FF')
        t4_2.scale(0.12)
        t4_2.move_to([-1.04, -0.68, 0])
        nums.add(t4_2)       

        tb4 = Tex(format_number(w[11], total_chars=6)).set_color('#FF00FF')
        tb4.scale(0.12)
        tb4.move_to([-1.04, -0.82, 0])
        nums.add(tb4)


        self.add(nums)


        self.wait()
        self.embed()

































