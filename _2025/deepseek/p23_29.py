from manimlib import *
from tqdm import tqdm
from pathlib import Path


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

from manimlib.mobject.svg.svg_mobject import _convert_point_to_3d
from manimlib.logger import log

def get_attention_head(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
                       svg_file='mha_2d_segments-',
                       img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'):

    q1=ImageMobject(str(img_path/'q_1.png'))
    q1.scale([0.0415, 0.08, 1]) 
    q1.move_to([-0.2,0.38,0]) 
    # self.add(q1)
    # self.remove(q1)      

    k1=ImageMobject(str(img_path/'k_1.png'))
    k1.scale([0.0415, 0.08, 1]) 
    k1.move_to([-0.2,-0.06,0]) 
    # self.add(k1)
    # self.remove(k1)

    v1=ImageMobject(str(img_path/'v_1.png'))
    v1.scale([0.0415, 0.08, 1]) 
    v1.move_to([-0.2,-0.48,0]) 
    # self.add(v1)
    # self.remove(v1)

    kt=ImageMobject(str(img_path/'k_1.png'))
    kt.scale([0.0215,0.035, 1])
    kt.rotate([0, 0, -PI/2]) 
    kt.move_to([0.405,0.305,0])     
    # self.add(kt)
    # self.remove(kt)

    a1=ImageMobject(str(img_path/'attention_scores.png'))
    a1.scale([0.055,0.055, 1])
    a1.move_to([0.66,0.37,0]) 
    # self.add(a1)
    # self.remove(a1)

    a2=ImageMobject(str(img_path/'attention_pattern.png'))
    a2.scale([0.13,0.13, 1])
    a2.move_to([1.27,0.25,0]) 
    # self.add(a2)
    # self.remove(a2)

    z1=ImageMobject(str(img_path/'z_1.png'))
    z1.scale([0.0425, 0.08, 1]) 
    z1.move_to([1.035,-0.48,0]) 
    # self.add(z1)
    # self.remove(z1)

    all_images=Group(q1, k1, v1, kt, a1, a2, z1)

    svg_files=list(sorted((svg_path).glob('*'+svg_file+'*')))
    # print(svg_files)

    all_labels=Group()
    for svg_file in svg_files:
        svg_image=SVGMobject(str(svg_file))
        all_labels.add(svg_image[1:]) #Thowout background
 
    large_white_connectors=SVGMobject(str(svg_path/'mha_2d_large_white_connectors_2.svg'))

    return Group(all_images, all_labels, large_white_connectors[1:])

def get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics'):


    x1=ImageMobject(str(Path(img_path)/'input_1_1.png'))
    x1.scale([0.075,0.125, 1]) 
    x1.move_to([-1.45,-0.03,0])

    x2=ImageMobject(str(Path(img_path)/'input_2_1.png'))
    x2.scale([0.075,0.125, 1]) 
    x2.move_to([-1.225,-0.03,0])
    
    all_images=Group(x1, x2)
    svg_image=SVGMobject(str(Path(svg_path)/svg_file))
    x_labels_1=svg_image[1:45] 

    return Group(all_images,x_labels_1)


class p12_29(InteractiveScene):
    def construct(self):

        #Start with previous 9 token input, and propaage 10 input thorugh
        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))

        x=get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path=img_path/'gpt_2_attention_viz_5')

        self.frame.reorient(0, 0, 0, (-0.01, -0.04, 0.0), 2.01)
        self.wait()

        self.add(a[0], a[1][:13], a[1][17], x[0])

        #Ok, first zoom in and change input text
        self.play(self.frame.animate.reorient(0, 0, 0, (-1.06, -0.05, 0.0), 0.88), run_time=2)

        i=0
        a2=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments_10_input-',
                                    img_path=img_path/'gpt_2_attention_viz_5'/str(i))

        self.wait()

        # a[1][1][:-10].scale([1, 0.9, 1]).shift(0.025*UP)
        # x[0].scale([1, 0.9, 1]).shift(0.025*UP)

        self.play(a[1][1][:-12].animate.scale([1, 0.9, 1]).shift(0.025*UP),
                x[0].animate.scale([1, 0.9, 1]).shift(0.025*UP))
        blue_word=a2[1][1][-20:-16].scale([1, 0.9, 1])
        self.add(blue_word.shift(0.005*UP)) #"Blue"
        self.wait()

        x1n=ImageMobject(str(img_path/'gpt_2_attention_viz_5'/'input_1_1n.png'))
        x1n.scale([0.0095,0.013, 1]) 
        x1n.move_to([-1.4492,-0.24,0])
        # self.add(x1n)

        x2n=ImageMobject(str(img_path/'gpt_2_attention_viz_5'/'input_2_1n.png'))
        x2n.scale([0.0089,0.012, 1]) 
        x2n.move_to([-1.224,-0.239,0])
        # self.add(x2n)
        # self.remove(x2n)

        # self.wait()
        self.play(FadeIn(x1n), FadeIn(x2n), FadeOut(a[1][1][-1]), FadeIn(a2[1][1][-2:]))

        self.wait()




        img_path_full=img_path/'gpt_2_attention_viz_5'/str(i)
        q1n=ImageMobject(str(img_path_full/'q_1n.png'))
        q1n.scale([0.0415, 0.08, 1]) 
        q1n.move_to([-0.2,0.38,0]) 


        self.wait(20)
        self.embed()









