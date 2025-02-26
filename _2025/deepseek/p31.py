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

class p31(InteractiveScene):
    def construct(self):

        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')


        # a=get_attention_head(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_segments-')
        # self.add(a)
        # self.wait()

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # self.wait()

        self.frame.reorient(0, 86, 0, (-0.07, -0.06, 0.06), 2.39)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # x.move_to([-1.85,0,0]) #Shift input data to the left
        # x.rotate([PI/2,0,0], axis=RIGHT)
        # self.add(x)        

        # self.remove(x)

        attention_heads=Group()
        spacing=0.25
        for i in range(12): #Render in reverse order for occlusions
            a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                        img_path=img_path/'gpt_2_attention_viz_1'/str(i))
            a.rotate([PI/2,0,0], axis=RIGHT)
            a.move_to([0, spacing*i,0])
            # a.set_opacity(0.5)
            attention_heads.add(a)
        
        # attention_heads.set_opacity(0.9)

        for i in range(11, 0, -1):
            self.add(attention_heads[i][1][13].set_opacity(0.9)) #Arrows on right side
            # self.add(attention_heads[i][1][0].set_opacity(0.9)) #Outlines

            self.add(attention_heads[i][0][0].set_opacity(0.8)) #queries
            self.add(attention_heads[i][0][1].set_opacity(0.8)) #keys
            self.add(attention_heads[i][0][2].set_opacity(0.8)) #values

            ##should replace kt with same matrix in each head ->
            self.add(attention_heads[i][0][3:].set_opacity(0.8)) #attention patterns and output
            # self.add(attention_heads[i][2]) #Thick white lines

        # -- Now do first layer a little differently
        # self.add(attention_heads[0][0][0].set_opacity(0.4)) #Queries
        self.add(attention_heads[0][0][3:].set_opacity(0.8)) #attention patterns and output.
        self.add(attention_heads[0][1][13].set_opacity(0.9)) #Arrows on right side
        self.add(attention_heads[0][0][0].set_opacity(0.8)) #queries
        self.add(attention_heads[0][0][1].set_opacity(0.8)) #keys
        self.add(attention_heads[0][0][2].set_opacity(0.8)) #values
        self.add(attention_heads[0][1][3].set_opacity(1.0)) #Query Label
        self.add(attention_heads[0][1][4].set_opacity(1.0)) #Key Label
        self.add(attention_heads[0][1][5].set_opacity(1.0)) #Vakye Label

        #Opationally add labels/variable names on front head
        self.add(attention_heads[0][1][14].set_opacity(0.5)) #Yea I'm 50/50 on this - nice to have the option i think
        # self.add(attention_heads[0][2]) #Thick white lines

        ## -- So, I think a really good option would be showing the keys and values collapsing down into a single matrix
        ## -- and then add in the connector. 
        self.wait()
        self.frame.reorient(0, 83, 0, (0.37, -0.06, 0.01), 2.04)
        self.wait()

        #Option with camera move - ok yeah this looks nice. 
        self.play(*[attention_heads[i][0][0].animate.set_opacity(0.0) for i in range(12)]+
	       [attention_heads[0][1][3].animate.set_opacity(0.0)]+
 		   [attention_heads[i][0][1].animate.move_to([-0.6 ,  1.391,  -0.08 ]) for i in range(12)]+ #Keys [-0.53 ,  1.391,  0.06 ]
 		   [attention_heads[0][1][4].animate.move_to([-0.6 ,  1.391,  -0.24 ])]+                   #[-0.53 ,  1.391,  -0.10 ]
 		   [attention_heads[i][0][2].animate.move_to([-0.6 ,  1.391, -0.48]) for i in range(12)]+ #Values [-0.53 ,  1.391, -0.34]
 		   [attention_heads[0][1][5].animate.move_to([-0.6 ,  1.391,  -0.65 ])]+                  #[-0.53 ,  1.391,  -0.51 ]
 		   [self.frame.animate.reorient(-53, 71, 0, (-0.36, 0.7, 0.07), 1.81)],
	       run_time=3)

        #Let's fade in connectors here. 

        connector_1=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/thick_white_connector_1.svg')
        connector_1.scale([1.0, 1.39, 1])
        connector_1.move_to([0.2, 1.389, -0.11]) #0.05 #[-0.12, 1.375, -0.08]
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.07])
        # self.add(connector_1) 

        connector_1b=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/thick_white_connector_1.svg')
        connector_1b.scale([1.0, 1.39, 1])
        connector_1b.move_to([ 0.14 ,  1.386, -0.499 ])
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.18, 1.375, -0.47])
        # self.add(connector_1b)

        # for i in range(12): self.add(attention_heads[i][2])
        white_arrows=Group(*[attention_heads[i][2] for i in range(12)])

        self.play(FadeIn(connector_1b), FadeIn(connector_1), FadeIn(white_arrows), 
        		 self.frame.animate.reorient(-34, 69, 0, (0.04, 0.58, 0.09), 2.10))
        self.add(attention_heads[0][0][1]) #Occlusion
        self.add(attention_heads[0][0][2]) #Occlusion
        self.wait()


        # self.play(self.frame.animate.reorient(-15, 67, 0, (0.34, 0.45, -0.07), 2.10)) #Optional Extra motion

        # reorient(-34, 69, 0, (0.04, 0.58, 0.09), 2.10)


        #Camera move then collapse. 
        # self.play(self.frame.animate.reorient(-31, 81, 0, (0.21, 0.03, -0.04), 2.04))
        # self.wait()

        # self.play(*[attention_heads[i][0][0].animate.set_opacity(0.0) for i in range(12)]+
        # 	       [attention_heads[0][1][3].animate.set_opacity(0.0)]+
        #  		   [attention_heads[i][0][1].animate.move_to([-0.53 ,  1.391,  0.06 ]) for i in range(12)]+
        #  		   [attention_heads[0][1][4].animate.move_to([-0.53 ,  1.391,  -0.10 ])]+
        #  		   [attention_heads[i][0][2].animate.move_to([-0.53 ,  1.391, -0.34]) for i in range(12)]+
        #  		   [attention_heads[0][1][5].animate.move_to([-0.53 ,  1.391,  -0.51 ])],
        # 	       run_time=3)

        # #Maybe labels move along with the front patternas we go here?
        # for i in range(12): attention_heads[i][0][0].set_opacity(0.0)
        # attention_heads[0][1][3].set_opacity(0.0)

        # for i in range(12): attention_heads[i][0][1].move_to([-0.53 ,  1.391,  0.06 ]) #Keys
        # attention_heads[0][1][4].move_to([-0.53 ,  1.391,  -0.10 ]) #Key labels


        # for i in range(12): attention_heads[i][0][2].move_to([-0.53 ,  1.391, -0.34]) #values
        # attention_heads[0][1][5].move_to([-0.53 ,  1.391,  -0.51 ]) #values labels


        # self.frame.reorient(-31, 81, 0, (0.21, 0.03, -0.04), 2.04)


        # self.play(self.frame.reorient(-31, 81, 0, (0.21, 0.03, -0.04), 2.04))





        connector_1=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/thick_white_connector_1.svg')
        connector_1.scale([1.0, 1.39, 1])
        connector_1.move_to([0.2, 1.386, -0.105]) #0.05 #[-0.12, 1.375, -0.08]
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.07])
        self.add(connector_1) 

        #Keys
        attention_heads[0][0][1].move_to([-0.53 ,  1.391,  0.06 ])
        self.add(attention_heads[0][0][1]) 
        self.add(attention_heads[0][1][4].move_to([-0.53 ,  1.391,  -0.10 ]))# Key labels

        connector_1b=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/thick_white_connector_1.svg')
        connector_1b.scale([1.0, 1.39, 1])
        connector_1b.move_to([ 0.14 ,  1.386, -0.495 ])
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.18, 1.375, -0.47])
        self.add(connector_1b)

        #Values
        # attention_heads[0][0][2].move_to([-0.53 ,  1.391, -0.34])
        # self.add(attention_heads[0][0][2]) 
        # self.add(attention_heads[0][1][5].move_to([-0.53 ,  1.391,  -0.51 ]))# Values labels
        # (np.array([0.2, 1.386, 0.05])-np.array([-0.12, 1.375, -0.08]))+np.array([-0.85, 1.38, -0.47])

        # head_connector=SVGMobject('/Users/stephen/welch_labs/deepseek/graphics/to_manim/head_connector_1.svg')
        # head_connector.scale([0.35, 1.385, 1])
        # head_connector.move_to([-1.45, 1.375, -0.047])
        # self.add(head_connector)
        # self.wait()
        
        # attention_heads[:-1].set_opacity(0.5)
        # self.add(attention_heads)






        


        self.wait(20)
        self.embed()



