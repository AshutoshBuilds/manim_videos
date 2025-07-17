from manimlib import *

 
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/backprop_3/graphics/'

def get_relu_joint(weight_1, weight_2, bias, extent=1):
    if np.abs(weight_2) < 1e-8: 
        x_intercept = -bias / weight_1
        return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
    elif np.abs(weight_1) < 1e-8:
        y_intercept = -bias / weight_2
        return [[-extent, y_intercept], [extent, y_intercept]] if -extent <= y_intercept <= extent else []
    else:
        points = []
        for x in [-extent, extent]:
            y = (-x * weight_1 - bias) / weight_2
            if -extent <= y <= extent: points.append([x, y])
        for y in [-extent, extent]:
            x = (-y * weight_2 - bias) / weight_1
            if -extent <= x <= extent: points.append([x, y])
        unique_points = []
        for p in points:
            is_duplicate = False
            for existing in unique_points:
                if abs(p[0] - existing[0]) < 1e-8 and abs(p[1] - existing[1]) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        return unique_points


class plane_folding_sketch_1(InteractiveScene):
    def construct(self):

        #nice trained 2 hidden layer model. 
        w1=np.array([[-0.02866297,  1.6250265 ],
             [-1.3056537 ,  0.46831134]], dtype=np.float32)
        b1=np.array([-0.4677289,  1.0067637], dtype=np.float32)
        w2=np.array([[ 1.3398709 ,  0.68694556],
                     [-0.29886743, -1.8411286 ]], dtype=np.float32)
        b2=np.array([-0.7817721 ,  0.90856946], dtype=np.float32)
        w3=np.array([[ 1.8897862,  3.0432484],
                     [-1.7220999, -2.2057745]], dtype=np.float32)
        b3=np.array([-1.0249746 ,  0.61326534], dtype=np.float32)
        
        map_img=ImageMobject(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.set_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.set_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(ORIGIN)

        self.frame.reorient(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        # self.add(map_img)
        # self.wait()


        def surface_func(u, v):
            # u and v are the parameters (from -1 to 1 for both)
            x = u
            y = v
            
            # Apply first layer transformation: w1*x + w2*y + b1
            linear_output = w1[0,0] * x + w1[0,1] * y + b1[0]
            
            # Apply ReLU activation
            relu_output = max(0, linear_output)
            
            # Use relu_output as the z-coordinate (height)
            z = relu_output * 0.5  # Scale down for better visualization
            
            return np.array([x, y, z])

        # Create the bent surface
        bent_surface = ParametricSurface(
            surface_func,
            u_range=[-1, 1],
            v_range=[-1, 1],
            resolution=(50, 50),
            # fill_color=BLUE,
            # fill_opacity=0.7,
            # stroke_color=WHITE,
            # stroke_width=1
        )
        ts=TexturedSurface(bent_surface, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        ts.set_shading(0,0,0)
        ts.set_opacity(0.95)

        self.add(ts)
        self.wait()

        # Ok great, bending looks good. And i think oreintation is actually mayby right?
        # Contour lines or heatmaps could be nice -> I think for now just a fold line would be a good starting point
        joint_points = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        
        if joint_points:
            # Create 3D points for the joint line
            joint_3d_points = []
            for point in joint_points:
                x, y = point
                # The z-coordinate at the joint is 0 (since ReLU output is 0 at the boundary)
                z = 0
                joint_3d_points.append([x, y, z])
            
            # Create a line connecting the joint points
            if len(joint_3d_points) >= 2:
                joint_line = DashedLine(
                    start=[joint_points[0][0], joint_points[0][1], 0],
                    end=[joint_points[1][0], joint_points[1][1], 0],
                    color=WHITE,
                    stroke_width=3,
                    dash_length=0.05
                )
                self.add(joint_line)
                


        self.wait()


        # self.add(bent_surface)
        
        # Move camera to 3D view
        # self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES)
        
        # Animate the transformation from flat to bent
        # self.play(
        #     Transform(map_img, bent_surface),
        #     run_time=3
        # )

        self.wait(20)
        self.embed()











