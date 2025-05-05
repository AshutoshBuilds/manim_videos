from manimlib import *

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

asset_dir='/Users/stephen/Stephencwelch Dropbox/Stephen Welch/welch_labs/backpropagation/graphics/'

class P6_8(InteractiveScene):
    def construct(self):


        v=SVGMobject(asset_dir+'intro_rewrite_graphics_2.svg')


        img_1=ImageMobject(asset_dir+'intro_rewrite_graphics_3.png')
        img_2=ImageMobject(asset_dir+'intro_rewrite_graphics_4.png')

        img_1.set_opacity(0.0)
        img_2.set_opacity(0.0)
        v.scale(2.0)
        self.add(img_1)
        self.add(img_2)

        self.frame.reorient(0, 0, 0, (-1.73, 0.07, 0.0), 2.35)
        
        self.play(ShowCreation(v[24])) #Left border
        self.wait()

        self.play(ShowCreation(v[2:14]))

        self.add(v[2:14]) #Part 1 label
        self.add(v[14:24]) #Part 2 label

        r1=RoundedRectangle(corner_radius=0.05, height=9.0/4.9, width=16.0/4.9)
        r1.set_stroke(color=BLUE) #CHILL_BROWN)
        r1.move_to([-1.7, 0, 0])
        self.add(r1)

        #Pick up here in the morning - I think i use manim rectangle class!


        self.wait()




        # self.remove(v[0])

        self.embed()
        self.wait(20)


