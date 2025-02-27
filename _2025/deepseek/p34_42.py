from manimlib import *
from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

from manimlib.mobject.svg.svg_mobject import _convert_point_to_3d
from manimlib.logger import log

def get_niave_mla_head(layer_id=0):


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

        # for l in all_labels:
        #     self.add(l)

        x1=ImageMobject(str(Path(input_image_path)/'input_1_1.png'))
        x1.scale([0.075,0.125, 1]) 
        x1.move_to([-1.47,0.155,0])
        # self.add(x1)
        # self.remove(x1)    

        x2=ImageMobject(str(Path(input_image_path)/'input_2_1.png'))
        x2.scale([0.075,0.125, 1]) 
        x2.move_to([-1.24,0.155,0])
        # self.add(x2)
        # self.remove(x2)        

        q1=ImageMobject(str(img_path/str(layer_id)/'q_nope.png'))
        q1.scale([0.0215, 0.078, 1]) 
        q1.move_to([-0.185,0.59,0]) 
        # self.add(q1)
        # self.remove(q1)      

        k1=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        k1.scale([0.0218, 0.078, 1]) 
        k1.move_to([-0.18,0.155,0]) 
        # self.add(k1)
        # self.remove(k1) 

        kt=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        kt.scale([0.0112, 0.03, 1])
        kt.rotate([0, 0, -PI/2]) 
        kt.move_to([0.425,0.5,0])     
        # self.add(kt)
        # self.remove(kt)

        v1=ImageMobject(str(img_path/str(layer_id)/'v_1.png'))
        v1.scale([0.0218, 0.078, 1]) 
        v1.move_to([-0.18,-0.27,0]) 
        # self.add(v1)
        # self.remove(v1)      

        kv=ImageMobject(str(img_path/'kv2.png'))
        kv.scale([0.0111, 0.115, 1]) 
        kv.move_to([-1.095,-0.735,0]) 
        # self.add(kv)
        # self.remove(kv)

        a1=ImageMobject(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.69,0.57,0]) 
        # self.add(a1)
        # self.remove(a1)

        a2=ImageMobject(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.28,0.445,0]) 
        # self.add(a2)
        # self.remove(a2)

        z1=ImageMobject(str(img_path/str(layer_id)/'x_1.png'))
        z1.scale([0.022, 0.07, 1]) 
        z1.move_to([1.055,-0.28,0]) 
        # self.add(z1)
        # self.remove(z1)

        all_images=Group(x1, x2, kv, q1, k1, kt, v1, a1, a2, z1)

        return Group(all_images, all_labels)



class p34_38(InteractiveScene):
    def construct(self):
        '''This sequence and the next will be a bit tricky, but I'll get some good mileage out of them 
           and then be home free. With this first one, I do like the idea of startiwth full 3d Niave MLA,
           and landing on 2d view to really explain it. Will require "bringing forward" the KV cache to 
           the 3d flat view and I think adding back in X (and probably queries) when I land on 2d space.'''

        # self.frame.reorient(0, 0, 0, (-0.12, 0.02, 0.0), 2.36)
        
        # a=get_niave_mla_head(0)
        # self.add(a)

        #Ok great, now what elements do we want in the 3d view?
        # elements_3d=Group(a[0][3:], a[1][4:13])
        # self.add(elements_3d)

        attention_heads=Group()
        spacing=0.25
        for i in range(12): 
            a=get_niave_mla_head(layer_id=6*i) #Step by 6 to get more variety
            a.rotate([PI/2,0,0], axis=RIGHT)
            a.move_to([0, spacing*i,0])
            attention_heads.add(a) #Group((a[0][3:], a[1][4:13])))

        for i in range(11, 0, -1): #Render in reverse order for occlusions
            self.add(attention_heads[i][0][3:].set_opacity(0.8)) #Images
            self.add(attention_heads[i][1][2:4]) #Flow chart
            self.add(attention_heads[i][1][11:13])
            self.add(attention_heads[i][1][14])
            self.add(attention_heads[i][1][4:6]) #Flow chart - thick white lines


        # --- Add some then all of first head info -> unclear when I fade out other heads yet
        self.add(attention_heads[0][0][3:].set_opacity(0.9)) #Images on right side
        self.add(attention_heads[0][1][2:4])
        self.add(attention_heads[0][1][6:17]) #Flow chart
        self.add(attention_heads[0][1][4:6])

        ## Now need single latents and connector - need latents to come forward aw se collapse down to 2D
        # Let's sstart with trying latents fully "beneath", move to to the left if that doesn't work. 
        #I may want to make my white connectors shorter?
        og_kv_cache_center=attention_heads[0][0][2].get_center().copy()
        self.add(attention_heads[0][0][2].move_to([-1.0780741 ,  1.38,  -0.714-0.15])) #KV Cache

        connector_1=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/medium_white_connector.svg')
        connector_1.rotate(PI/2, DOWN)
        connector_1.scale([0.4, 1.39, 1])
        connector_1.move_to([ -0.99 ,  1.38, -0.56 ])
        self.add(connector_1)

        # self.remove(connector_1)

        self.frame.reorient(-41, 72, 0, (-0.28, 0.77, -0.13), 2.98)
        # self.frame.reorient(-37, 67, 0, (-0.84, 1.04, -0.56), 2.47) #Option to start more zoomed in on kv cache
        self.add(attention_heads[0][1][4:6]) #These aren't sticking on top for some reaon, add again. 
        self.wait()

        # Now pan camera to front while bringing KV cache forward and fading out think white connector - maybe leaving behind 
        # more chill brown ones?

        self.play(*[FadeOut(attention_heads[i][1][4:6]) for i in range(12)]+
                   [FadeOut(connector_1)]+
                   [attention_heads[0][0][2].animate.move_to(og_kv_cache_center)]+
                   [self.frame.animate.reorient(0, 90, 0, (-0.05, 0.79, -0.04), 2.70)], #Placeholder position will need to tweak
                   run_time=4) #Do an option with cranked up runtime - this covers a couple paragraphs
        
        for i in range(1, 12): self.remove(attention_heads[i])
        self.add(attention_heads[0][0][:2])
        self.add(attention_heads[0][1][:2])

        self.wait()

        # self.frame.reorient(0, 89, 0, (-0.05, 0.79, -0.04), 2.67)


        self.wait(20)
        self.embed()



class niave_mla_hacking(InteractiveScene):
    def construct(self):
        '''This sequence and the next will be a bit tricky, but I'll get some good mileage out of them 
           and then be home free. With this first one, I do like the idea of startiwth full 3d Niave MLA,
           and landing on 2d view to really explain it. Will require "bringing forward" the KV cache to 
           the 3d flat view and I think adding back in X (and probably queries) when I land on 2d space.'''

        # niave_mla_panels-01.svg
        # Ok I don't have 3d naive MLA preped, so first this is to probably build the 2d version live here
        # and then roll up into a support function

        self.frame.reorient(0, 0, 0, (-0.12, 0.02, 0.0), 2.36)

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
        x1.move_to([-1.47,0.155,0])
        self.add(x1)
        # self.remove(x1)    

        x2=ImageMobject(str(Path(input_image_path)/'input_2_1.png'))
        x2.scale([0.075,0.125, 1]) 
        x2.move_to([-1.24,0.155,0])
        self.add(x2)
        # self.remove(x2)        

        q1=ImageMobject(str(img_path/str(layer_id)/'q_nope.png'))
        q1.scale([0.0215, 0.078, 1]) 
        q1.move_to([-0.185,0.59,0]) 
        self.add(q1)
        # self.remove(q1)      

        k1=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        k1.scale([0.0218, 0.078, 1]) 
        k1.move_to([-0.18,0.155,0]) 
        self.add(k1)
        # self.remove(k1) 

        kt=ImageMobject(str(img_path/str(layer_id)/'k_1.png'))
        kt.scale([0.0112, 0.03, 1])
        kt.rotate([0, 0, -PI/2]) 
        kt.move_to([0.425,0.5,0])     
        self.add(kt)
        self.remove(kt)

        v1=ImageMobject(str(img_path/str(layer_id)/'v_1.png'))
        v1.scale([0.0218, 0.078, 1]) 
        v1.move_to([-0.18,-0.27,0]) 
        self.add(v1)
        # self.remove(v1)      

        kv=ImageMobject(str(img_path/'kv2.png'))
        kv.scale([0.0111, 0.115, 1]) 
        kv.move_to([-1.095,-0.735,0]) 
        self.add(kv)
        # self.remove(kv)

        a1=ImageMobject(str(img_path/str(layer_id)/'scores_single_1.png'))
        a1.scale([0.055,0.055, 1])
        a1.move_to([0.69,0.57,0]) 
        self.add(a1)
        # self.remove(a1)

        a2=ImageMobject(str(img_path/str(layer_id)/'scores_single_4.png'))
        a2.scale([0.13,0.13, 1])
        a2.move_to([1.28,0.445,0]) 
        self.add(a2)
        # self.remove(a2)

        z1=ImageMobject(str(img_path/str(layer_id)/'x_1.png'))
        z1.scale([0.022, 0.07, 1]) 
        z1.move_to([1.055,-0.28,0]) 
        self.add(z1)
        # self.remove(z1)

        all_images=Group(x1, x2, q1, k1, kt, v1, kv, a1, a2, z1)







        self.wait(20)
        self.embed()