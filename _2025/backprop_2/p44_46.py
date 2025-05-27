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
heatmap_path='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backprop2/graphics/to_manim/may_27_1'

# map_min_x=0.38
# map_max_x=1.54
# map_min_y=-0.56
# map_max_y=0.56

# min_long=-7.0
# max_long=18.0
# min_lat=36.0
# max_lat=56.0


def format_number(num, total_chars=5):
    """
    Format number to always show same total character width including sign.
    
    Args:
        num: The number to format
        total_chars: Total character width including decimal point and sign
    """
    abs_num = abs(num)
    sign = '-' if num < 0 else ''
    
    # Account for sign in character count
    available_chars = total_chars - len(sign)
    
    if abs_num >= 1000:
        # For very large numbers, use scientific notation or integer
        formatted = f"{num:.0f}"
    elif abs_num >= 100:
        # 100-999: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: adjust decimal places based on available space
        if available_chars >= 4:  # Room for "0.xx"
            formatted = f"{num:.2f}"
        else:
            formatted = f"{num:.1f}"
    
    return formatted


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

def get_numbers(i, xs, weights, logits, yhats):
    x=xs[i, -1]

    numbers=VGroup()


    tx = Tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    numbers.add(tx)

    w=weights[i,:]
    tm1=Tex(format_number(w[0])).set_color('#00FFFF')
    tm1.scale(0.16)
    tm1.move_to([-1.195, 0.205, 0])
    numbers.add(tm1)

    tm2=Tex(format_number(w[1])).set_color(YELLOW)
    tm2.scale(0.15)
    tm2.move_to([-1.155, 0.015, 0])
    numbers.add(tm2)

    tm3=Tex(format_number(w[2])).set_color(GREEN)
    tm3.scale(0.16)
    tm3.move_to([-1.19, -0.17, 0])
    numbers.add(tm3)

    tb1=Tex(format_number(w[3])).set_color('#00FFFF')
    tb1.scale(0.16)
    tb1.move_to([-0.875, 0.365, 0])
    numbers.add(tb1)

    tb2=Tex(format_number(w[4])).set_color(YELLOW)
    tb2.scale(0.16)
    tb2.move_to([-0.875, 0.015, 0])
    numbers.add(tb2)

    tb3=Tex(format_number(w[5])).set_color(GREEN)
    tb3.scale(0.16)
    tb3.move_to([-0.88, -0.335, 0])
    numbers.add(tb3)

    tl1=Tex(format_number(logits[i,0])).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    numbers.add(tl1)

    tl2=Tex(format_number(logits[i,1])).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    numbers.add(tl2)

    tl3=Tex(format_number(logits[i,2])).set_color(GREEN)
    tl3.scale(0.16)
    tl3.move_to([-0.52, -0.335, 0])
    numbers.add(tl3)

    yhat1=Tex(format_number(yhats[i,0])).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)

    yhat2=Tex(format_number(yhats[i,1])).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)

    yhat3=Tex(format_number(yhats[i,2])).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    return numbers

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



class p44_v2(InteractiveScene):
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

















