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



class p12_20(InteractiveScene):
    def construct(self):

        img_path=Path('/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek')
        svg_path=Path('/Users/stephen/welch_labs/deepseek/graphics/to_manim')

        i=0
        a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
                                    img_path=img_path/'gpt_2_attention_viz_2'/str(i))

        x=get_input_x(svg_path='/Users/stephen/welch_labs/deepseek/graphics/to_manim',
           svg_file='mha_2d_grouping_test.svg',
           img_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/flowchart_graphics')

        self.frame.reorient(0, 0, 0, (-1.39, -0.06, -0.0), 1.16)
        self.wait()
        # self.add(x[0][0])
        # self.add(x[0][1])

        self.play(FadeIn(x[0]))
        self.add(a[1][1]) #xlabels
        self.add(a[1][2]) #xlabels
        self.wait()

        self.add(a[1][16]) #deepseek dim
        self.wait()
        self.remove(a[1][16])
        self.wait()

        self.play(FadeIn(a[1][6]), self.frame.animate.reorient(0, 0, 0, (-0.65, 0.02, 0.0), 1.36), run_time=2)
        self.wait()

        self.play(FadeIn(a[1][3]), FadeIn(a[1][4]), FadeIn(a[0][0]), FadeIn(a[0][1]))
        self.wait()

        # self.add(a[1][3])
        # self.add(a[1][4])
        # Can i, withoug going insane, actually have these matrices be made up of rows and 
        # then break them apart and pull one out of each?

        #Ok hack on row by row version here then break into subfuncrion


        separate_row_im_path='/Users/stephen/welch_labs/deepseek/hackin/linux_workdir/deepseek/gpt_2_attention_viz_3'
        q_rows=Group()
        for row_id in range(9):
            q=ImageMobject(separate_row_im_path+'/query_row_'+str(row_id)+'.png')
            q.scale([0.0127, 0.024, 1]) 
            q.move_to([-0.2,0.492-0.028*row_id,0]) 
            q_rows.add(q)
        self.add(q_rows)
        # self.remove(q_rows)

        k_rows=Group()
        for row_id in range(9):
            k=ImageMobject(separate_row_im_path+'/key_row_'+str(row_id)+'.png')
            k.scale([0.0127, 0.024, 1]) 
            k.move_to([-0.2,0.1-0.028*row_id,0]) 
            k_rows.add(k)
        self.add(k_rows)

        self.remove(k_rows)



        # q1=ImageMobject(str(img_path/'q_1.png'))
        # q1.scale([0.0415, 0.08, 1]) 
        # q1.move_to([-0.2,0.38,0]) 
        # # self.add(q1)
        # # self.remove(q1)      

        # k1=ImageMobject(str(img_path/'k_1.png'))
        # k1.scale([0.0415, 0.08, 1]) 
        # k1.move_to([-0.2,-0.06,0]) 



        self.wait()
        self.embed()

        #         attention_heads=Group()
        # spacing=0.25
        # for i in range(12): #Render in reverse order for occlusions
        #     a=get_attention_head(svg_path=svg_path,svg_file='mha_2d_segments-',
        #                                 img_path=img_path/'gpt_2_attention_viz_1'/str(i))
        #     a.rotate([PI/2,0,0], axis=RIGHT)
        #     a.move_to([0, spacing*i,0])
        #     # a.set_opacity(0.5)
        #     attention_heads.add(a)


        # a=get_attention_head_course(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1/0', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(a)

        # x=get_input_x(svg_path=svg_path, img_path=img_path/'gpt_2_attention_viz_1', 
        #                             svg_file='mha_2d_grouped_separate_lines.svg')
        # self.add(x)

        # self.frame.reorient(0, 0, 0, (0.00, 0.00, 0.00), 2.80)
        # # self.wait()



