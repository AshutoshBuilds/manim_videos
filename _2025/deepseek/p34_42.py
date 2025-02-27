from manimlib import *
from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

from manimlib.mobject.svg.svg_mobject import _convert_point_to_3d
from manimlib.logger import log


class p34_38(InteractiveScene):
    def construct(self):
        '''This sequence and the next will be a bit tricky, but I'll get some good mileage out of them 
           and then be home free. With this first one, I do like the idea of startiwth full 3d Niave MLA,
           and landing on 2d view to really explain it. Will require "bringing forward" the KV cache to 
           the 3d flat view and I think adding back in X (and probably queries) when I land on 2d space.'''

        # niave_mla_panels-01.svg
        # Ok I don't have 3d naive MLA preped, so first this is to probably build the 2d version live here
        # and then roll up into a support function



        layer_id=0

        input_image_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')
        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/deepseek_viz_2')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')
        svg_file='niave_mla_panels-'

        # self.frame.reorient(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)
        # self.frame.reorient(0, 0, 0, (0.58, 0.21, 0.0), 1.86)

        svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
        # print(svg_files)

        all_labels=Group()
        for svg_file in svg_files[:-1]: #Dont bring in last panel
            svg_image=SVGMobject(str(svg_file))
            all_labels.add(svg_image[1:]) #Thowout background

        for l in all_labels:
            self.add(l)

            

        x1=ImageMobject(str(Path(input_image_path)/'input_1_1.png'))
        x1.scale([0.075,0.125, 1]) 
        x1.move_to([-1.45,-0.03,0])
        # self.add(x1)
        # self.remove(x1)    

        x2=ImageMobject(str(Path(input_image_path)/'input_2_1.png'))
        x2.scale([0.075,0.125, 1]) 
        x2.move_to([-1.225,-0.03,0])
        # self.add(x2)
        # self.remove(x2)        

        q1=ImageMobject(str(img_path/str(layer_id)/'q_1.png'))
        q1.scale([0.0075, 0.14, 1]) 
        q1.move_to([-0.285,0.6,0]) 
        # self.add(q1)
        # self.remove(q1)      

        k1=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        k1.scale([0.0075, 0.14, 1]) 
        k1.move_to([-0.285,0.6,0]) 
        # self.add(k1)
        # self.remove(k1) 

        kt=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        kt.scale([0.004, 0.045, 1])
        kt.rotate([0, 0, -PI/2]) 
        kt.move_to([0.293,0.485,0])     
        # self.add(kt)
        # self.remove(kt)

        v1=ImageMobject(str(img_path/str(layer_id)/'v_1.png'))
        v1.scale([0.0075, 0.14, 1]) 
        v1.move_to([-0.285,0.6,0]) 
        # self.add(v1)
        # self.remove(v1)      

        kv=ImageMobject(str(img_path/'kv2.png'))
        kv.scale([0.0075, 0.14, 1]) 
        kv.move_to([-0.285,-0.01,0]) 
        # self.add(kv)
        # self.remove(kv)

        a1=ImageMobject(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.56,0.56,0]) 
        # self.add(a1)
        # self.remove(a1)

        a2=ImageMobject(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.155,0.445,0]) 
        # self.add(a2)
        # self.remove(a2)

        z1=ImageMobject(str(img_path/str(layer_id)/'x_1.png'))
        z1.scale([0.0079, 0.10, 1]) 
        z1.move_to([0.91,-0.2,0]) 
        # self.add(z1)
        # self.remove(z1)

        all_images=Group(x1, x2, q1, k1, kt, v1, kv, a1, a2, z1)







        self.wait(20)
        self.embed()