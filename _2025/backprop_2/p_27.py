from manimlib import *

CHILL_BROWN='#cabba6'
COOL_BLUE = '#00ffff'
COOL_YELLOW = '#ffd35a'
COOL_GREEN = '#00a14b'

class P27(InteractiveScene):
    def construct(self):
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq.set_color_by_tex("h_1", COOL_BLUE)
        softmax_eq.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq.set_color_by_tex("h_3", COOL_GREEN)
        
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2.set_color_by_tex("h_1", COOL_BLUE)
        softmax_eq_2.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq_2.set_color_by_tex("h_3", COOL_GREEN)
        softmax_eq_2.submobjects[7].set_color(WHITE)
        softmax_eq_2.submobjects[5].set_color(COOL_YELLOW)

        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")

        # Position the initial equations
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN).move_to(ORIGIN)

        # Store the original positions before any transformations
        softmax_original_pos = softmax_eq.get_center()
        cross_entropy_original_pos = cross_entropy_loss_eq.get_center()

        # Position the target equations to match exactly
        softmax_eq_2.move_to(softmax_original_pos)
        cross_entropy_loss_eq_2.move_to(cross_entropy_original_pos)

        softmax_label = Text("Softmax Function", weight=BOLD, font_size=24).set_color(CHILL_BROWN).next_to(softmax_eq, RIGHT * 8)

        cross_entropy_loss_label = Text("Cross-Entropy Loss", weight=BOLD, font_size=24).set_color(CHILL_BROWN)
        cross_entropy_loss_label.move_to(cross_entropy_loss_eq.get_center()).shift(RIGHT * (softmax_label.get_right()[0] - cross_entropy_loss_label.get_right()[0]))

        center_eqs_labels = VGroup(softmax_label, cross_entropy_loss_label)

        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(COOL_BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(COOL_YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(COOL_GREEN)

        h_group = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).next_to(center_eqs, LEFT * 5)

        # Create all the subsequent transformation equations
        softmax_eq_3 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}}")
        for i in [5, 6, 17, 18]:
            softmax_eq_3.submobjects[i].set_color(COOL_YELLOW)
        for i in [9, 10, 11, 12, 13, 14]:
            softmax_eq_3.submobjects[i].set_color(COOL_BLUE)
        for i in [21, 22]:
            softmax_eq_3.submobjects[i].set_color(COOL_GREEN)
            
        softmax_eq_4 = Tex(r"\hat{y}_2 = \frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}}")
        for i in [5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26]:
            softmax_eq_4.submobjects[i].set_color(COOL_YELLOW)
        for i in [13, 14, 15, 16, 17, 18]:
            softmax_eq_4.submobjects[i].set_color(COOL_BLUE)
        for i in [29, 30]:
            softmax_eq_4.submobjects[i].set_color(COOL_GREEN)
            
        softmax_eq_5 = Tex(r"\hat{y}_2 = \frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}}")
        for i in [5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26]:
            softmax_eq_5.submobjects[i].set_color(COOL_YELLOW)
        for i in [13, 14, 15, 16, 17, 18]:
            softmax_eq_5.submobjects[i].set_color(COOL_BLUE)
        for i in [29, 30, 31, 32, 33, 34]:
            softmax_eq_5.submobjects[i].set_color(COOL_GREEN)
        
        # Position all transformation equations to the same location
        softmax_eq_3.move_to(softmax_original_pos)
        softmax_eq_4.move_to(softmax_original_pos)
        softmax_eq_5.move_to(softmax_original_pos)
                
        self.play(Write(center_eqs), Write(center_eqs_labels), Write(h_group))

        # First transformation - ensure positions are locked
        self.play(
            ReplacementTransform(softmax_eq, softmax_eq_2),
            ReplacementTransform(cross_entropy_loss_eq, cross_entropy_loss_eq_2),
            Indicate(VGroup(softmax_eq_2.submobjects[2], softmax_eq_2.submobjects[6])),
            Indicate(cross_entropy_loss_eq_2.get_parts_by_tex("i"))
        )
        
        # After transformation, ensure positions are still correct
        softmax_eq_2.move_to(softmax_original_pos)
        cross_entropy_loss_eq_2.move_to(cross_entropy_original_pos)
        
        self.play(FlashUnder(softmax_eq_2[2]), FlashUnder(softmax_eq_2[6]), FlashUnder(cross_entropy_loss_eq_2[11]))
        
        self.remove(softmax_eq)
        self.remove(cross_entropy_loss_eq)

        self.play(
            FlashAround(h1_eq.get_part_by_tex("h_1"), color=COOL_BLUE),
            FlashAround(softmax_eq_2.get_part_by_tex("h_1"), color=COOL_BLUE),
            run_time=1.25
        )

        # Before transformation, ensure target is in correct position
        softmax_eq_3.move_to(softmax_original_pos)
        
        self.play(
            ReplacementTransform(
                VGroup(h1_eq.get_part_by_tex("m_1 x + b_1"), softmax_eq_2),
                softmax_eq_3
            ),
            Uncreate(h1_eq.get_part_by_tex("h_1 = ")),
            Uncreate(center_eqs_labels),
            run_time=1.25
        )
        
        # Ensure position after transformation
        softmax_eq_3.move_to(softmax_original_pos)
        
        self.play(
            FlashAround(h2_eq.get_part_by_tex("h_2"), color=COOL_YELLOW),
            FlashAround(VGroup(softmax_eq_3.submobjects[5], softmax_eq_3.submobjects[6]), color=COOL_YELLOW),
            FlashAround(VGroup(softmax_eq_3[17], softmax_eq_3[18]), color=COOL_YELLOW),
            run_time=1.25
        )

        # Position target before transformation
        softmax_eq_4.move_to(softmax_original_pos)
        
        self.play(
            ReplacementTransform(
                VGroup(h2_eq.get_part_by_tex("m_2 x + b_2"), softmax_eq_3),
                softmax_eq_4
            ),
            Uncreate(h2_eq.get_part_by_tex("h_2 = ")),
            run_time=1.25
        )
        
        # Ensure position after transformation
        softmax_eq_4.move_to(softmax_original_pos)

        self.play(
            FlashAround(h3_eq.get_part_by_tex("h_3"), color=COOL_GREEN),
            FlashAround(softmax_eq_4.get_part_by_tex("h_3"), color=COOL_GREEN),
            run_time=1.25
        )

        # Position target before transformation
        softmax_eq_5.move_to(softmax_original_pos)
        
        self.play(
            ReplacementTransform(
                VGroup(h3_eq.get_part_by_tex("m_3 x + b_3"), softmax_eq_4),
                softmax_eq_5
            ),
            Uncreate(h3_eq.get_part_by_tex("h_3 = ")),
            run_time=1.25
        )
        
        # Ensure final position
        softmax_eq_5.move_to(softmax_original_pos)
        cross_entropy_loss_eq_2.move_to(cross_entropy_original_pos)
        
        self.play(
            FlashAround(VGroup(softmax_eq_5.submobjects[0], softmax_eq_5.submobjects[1], softmax_eq_5.submobjects[2]), color=WHITE), 
            FlashAround(VGroup(cross_entropy_loss_eq_2.submobjects[9], cross_entropy_loss_eq_2.submobjects[10], cross_entropy_loss_eq_2.submobjects[11]), color=WHITE)
        )
        
        self.play(
            FadeOut(VGroup(softmax_eq_5.submobjects[0], softmax_eq_5.submobjects[1], softmax_eq_5.submobjects[2])), 
            FadeOut(VGroup(cross_entropy_loss_eq_2.submobjects[9], cross_entropy_loss_eq_2.submobjects[10], cross_entropy_loss_eq_2.submobjects[11])), 
            FadeOut(softmax_eq_5.submobjects[3])
        )
        
        target_x = softmax_eq_5.submobjects[11].get_left()[0]
        
        submob8 = cross_entropy_loss_eq_2.submobjects[8]
        shift_amount = target_x - submob8.get_right()[0]
        left_side = VGroup(*[cross_entropy_loss_eq_2.submobjects[i] for i in range(8)])
        
        right_target_x = softmax_eq_5.submobjects[11].get_right()[0]
        right_shift_amount = right_target_x - cross_entropy_loss_eq_2.submobjects[12].get_left()[0]
        softmax_eq_5_slice = VGroup(*softmax_eq_5.submobjects[4:35])
        down = softmax_eq_5_slice.get_center()[1] - submob8.get_center()[1]
        
        self.play(
            submob8.animate.shift(RIGHT * shift_amount), 
            left_side.animate.shift(RIGHT * shift_amount), 
            cross_entropy_loss_eq_2.submobjects[12].animate.shift(RIGHT * right_shift_amount),
            softmax_eq_5_slice.animate.shift(DOWN * down)
        )

        self.embed()      
        
class P27V2(InteractiveScene):
    def construct(self):
        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(COOL_BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(COOL_YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(COOL_GREEN)
        
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq.set_color_by_tex("h_1", COOL_BLUE)
        softmax_eq.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq.set_color_by_tex("h_3", COOL_GREEN)
        
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2.set_color_by_tex("h_1", COOL_BLUE)
        softmax_eq_2.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq_2.set_color_by_tex("h_3", COOL_GREEN)
        softmax_eq_2.submobjects[7].set_color(WHITE)
        softmax_eq_2.submobjects[5].set_color(COOL_YELLOW)
        
        softmax_eq_3 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_3.set_color_by_tex("m_1 x + b_1", COOL_BLUE)
        softmax_eq_3.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq_3.set_color_by_tex("h_3", COOL_GREEN)
        softmax_eq_3.submobjects[7].set_color(WHITE)
        softmax_eq_3.submobjects[5].set_color(COOL_YELLOW)
        
        softmax_eq_4 = Tex(r"\hat{y}_2 = \frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}}")
        softmax_eq_4.set_color_by_tex("m_1 x + b_1", COOL_BLUE)
        softmax_eq_4.set_color_by_tex("m_2 x + b_2", COOL_YELLOW)
        softmax_eq_4.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq_4.set_color_by_tex("h_3", COOL_GREEN)
        softmax_eq_4.submobjects[5].set_color(COOL_YELLOW)
        softmax_eq_4.submobjects[11].set_color(WHITE)
        
        softmax_eq_5 = Tex(r"\hat{y}_2 = \frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}}")
        softmax_eq_5.set_color_by_tex("m_1 x + b_1", COOL_BLUE)
        softmax_eq_5.set_color_by_tex("m_2 x + b_2", COOL_YELLOW)
        softmax_eq_5.set_color_by_tex("m_3 x + b_3", COOL_GREEN)
        softmax_eq_5.set_color_by_tex("h_2", COOL_YELLOW)
        softmax_eq_5.set_color_by_tex("h_3", COOL_GREEN)
        softmax_eq_5.submobjects[5].set_color(COOL_YELLOW)
        softmax_eq_5.submobjects[11].set_color(WHITE)
        
        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")
        
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN).move_to(ORIGIN)
        h_eqs_group = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).next_to(center_eqs, LEFT * 5)
        
        softmax_eq_2.move_to(softmax_eq.get_center())
        softmax_eq_3.move_to(softmax_eq.get_center())
        softmax_eq_4.move_to(softmax_eq.get_center())
        softmax_eq_5.move_to(softmax_eq.get_center())
        cross_entropy_loss_eq_2.move_to(cross_entropy_loss_eq.get_center())
        
        cross_entropy_loss_eq_3 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}})")
        cross_entropy_loss_eq_3.move_to(cross_entropy_loss_eq.get_center())
        cross_entropy_loss_eq_3.set_color_by_tex("m_1 x + b_1", COOL_BLUE)
        cross_entropy_loss_eq_3.set_color_by_tex("m_2 x + b_2", COOL_YELLOW)
        cross_entropy_loss_eq_3.set_color_by_tex("m_3 x + b_3", COOL_GREEN)
        cross_entropy_loss_eq_3.submobjects[10].set_color(COOL_YELLOW)
        cross_entropy_loss_eq_3.submobjects[16].set_color(WHITE)
        cross_entropy_loss_eq_3.move_to(ORIGIN)
        
        self.play(Write(center_eqs), 
                  Write(h_eqs_group))
        
        '''self.play(Indicate(VGroup(softmax_eq[6], 
                                  softmax_eq[2], 
                                  cross_entropy_loss_eq[11])))'''
        
        self.wait()
        
        self.play(ReplacementTransform(softmax_eq[6], softmax_eq_2[6]), 
                  ReplacementTransform(softmax_eq[2], softmax_eq_2[2]), 
                  ReplacementTransform(cross_entropy_loss_eq[11], cross_entropy_loss_eq_2[11]), 
                  ReplacementTransform(softmax_eq[5], softmax_eq_2[5]))
        
        self.wait()
        
        '''self.play(
            FlashAround(h1_eq.get_part_by_tex("h_1"), color=COOL_BLUE),
            FlashAround(softmax_eq.get_part_by_tex("h_1"), color=COOL_BLUE),
        )
        
        self.play(
            FlashAround(h2_eq.get_part_by_tex("h_2"), color=COOL_YELLOW),
            FlashAround(VGroup(softmax_eq_2.submobjects[5], softmax_eq_2.submobjects[6]), color=COOL_YELLOW),
            FlashAround(VGroup(softmax_eq_2.submobjects[13], softmax_eq_2.submobjects[14]), color=COOL_YELLOW),
        )
        
        self.play(
            FlashAround(h3_eq.get_part_by_tex("h_3"), color=COOL_GREEN),
            FlashAround(softmax_eq.get_part_by_tex("h_3"), color=COOL_GREEN),
        )'''
        
        self.wait()
                
        self.play(
            ReplacementTransform(softmax_eq[0:2], softmax_eq_3[0:2]),              # eq1 0 to 1 -> eq3 0 to 1
            ReplacementTransform(softmax_eq_2[2], softmax_eq_3[2]),                # eq2 2 -> eq3 2
            ReplacementTransform(softmax_eq[3], softmax_eq_3[3]),                  # eq1 3 -> eq3 3
            ReplacementTransform(softmax_eq[4], softmax_eq_3[4]),                  # eq1 4 -> eq3 4
            ReplacementTransform(softmax_eq_2[5:7], softmax_eq_3[5:7]),            # eq2 5 to 6 -> eq3 5 to 6
            ReplacementTransform(softmax_eq[7], softmax_eq_3[7]),                  # eq1 7 -> eq3 7
            ReplacementTransform(softmax_eq[8], softmax_eq_3[8]),                  # eq1 8 -> eq3 8
            ReplacementTransform(VGroup(h1_eq.get_part_by_tex("m_1 x + b_1")), softmax_eq_3[9:15]),            # eq1 9 to 10 -> eq3 9 to 14,
            FadeOut(h1_eq.get_part_by_tex("h_1 = ")),
            ReplacementTransform(softmax_eq[11:19], softmax_eq_3[15:24]),
            FadeOut(softmax_eq[9:11])# eq1 11 to 18 -> eq3 15 to 23
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(softmax_eq_3[0:4], softmax_eq_4[0:4]),               # eq3 0,1,2,3 -> eq4 0,1,2,3
            ReplacementTransform(softmax_eq_3[4], softmax_eq_4[4]),                   # eq3 4 -> eq4 4
            ReplacementTransform(VGroup(h2_eq.get_part_by_tex("m_2 x + b_2")), softmax_eq_4[5:11]),              # eq3 5,6 -> eq4 5–10
            ReplacementTransform(softmax_eq_3[7], softmax_eq_4[11]),                  # eq3 7 -> eq4 11
            ReplacementTransform(softmax_eq_3[8], softmax_eq_4[12]),                  # eq3 8 -> eq4 12
            ReplacementTransform(softmax_eq_3[9:15], softmax_eq_4[13:19]),            # eq3 9–14 -> eq4 13–18
            ReplacementTransform(softmax_eq_3[15:17], softmax_eq_4[19:21]),           # eq3 15–16 -> eq4 19–20
            ReplacementTransform(VGroup((h2_eq.get_part_by_tex("m_2 x + b_2").copy())), softmax_eq_4[21:27]),           # eq3 17–18 -> eq4 21–26
            FadeOut(h2_eq.get_part_by_tex("h_2 = ")),
            ReplacementTransform(softmax_eq_3[19:23], softmax_eq_4[27:31]),
            FadeOut(softmax_eq_3[17:19]),# eq3 19–22 -> eq4 27–30,
            FadeOut(softmax_eq_3[5:7])
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(softmax_eq_4[0:29], softmax_eq_5[0:29]),
            ReplacementTransform(VGroup(h3_eq.get_part_by_tex("m_3 x + b_3")), softmax_eq_5[29:35]),# eq4 0–28 -> eq5 0–28   
            FadeOut(h3_eq.get_part_by_tex("h_3 = ")),
            FadeOut(softmax_eq_4[29:31])
        )
        
        self.wait()
                
        self.play(
            ReplacementTransform(cross_entropy_loss_eq[0:8], cross_entropy_loss_eq_3[0:8]),
            ReplacementTransform(cross_entropy_loss_eq[8], cross_entropy_loss_eq_3[8]),
            ReplacementTransform(cross_entropy_loss_eq[12], cross_entropy_loss_eq_3[40]),
            ReplacementTransform(softmax_eq_5[4:35], cross_entropy_loss_eq_3[9:40]),
            FadeOut(softmax_eq_5[0:4]),
            FadeOut(VGroup(cross_entropy_loss_eq[9:11], cross_entropy_loss_eq_2[11]))
        )
        
        self.wait()
                
        
        self.embed()

class P27V3(InteractiveScene):
    def construct(self):
        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(COOL_BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(COOL_YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(COOL_GREEN)
        
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq[9:11].set_color(COOL_BLUE)
        softmax_eq[13:15].set_color(COOL_YELLOW)
        softmax_eq[17:19].set_color(COOL_GREEN)
        
        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2[5:7].set_color(COOL_YELLOW)
        softmax_eq_2[9:11].set_color(COOL_BLUE)
        softmax_eq_2[13:15].set_color(COOL_YELLOW)
        softmax_eq_2[17:19].set_color(COOL_GREEN)
        
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")
        
        big_eq = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}})")
        big_eq[10:12].set_color(COOL_YELLOW)
        big_eq[14:16].set_color(COOL_BLUE)
        big_eq[18:20].set_color(COOL_YELLOW)
        big_eq[22:24].set_color(COOL_GREEN)
        
        big_eq_2 = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}})")
        big_eq_2[10:12].set_color(COOL_YELLOW)
        big_eq_2[14:20].set_color(COOL_BLUE)
        big_eq_2[22:24].set_color(COOL_YELLOW)
        big_eq_2[26:28].set_color(COOL_GREEN)
        
        
        big_eq_3 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}})")
        big_eq_3[10:16].set_color(COOL_YELLOW)
        big_eq_3[18:24].set_color(COOL_BLUE)
        big_eq_3[26:32].set_color(COOL_YELLOW)
        big_eq_3[34:36].set_color(COOL_GREEN)
        
        big_eq_4 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}})")
        big_eq_4[10:16].set_color(COOL_YELLOW)
        big_eq_4[18:24].set_color(COOL_BLUE)
        big_eq_4[26:32].set_color(COOL_YELLOW)
        big_eq_4[34:40].set_color(COOL_GREEN)
        
        final_eq = Tex(
            r"\frac{\partial Loss}{\partial m_2} = \frac{\partial}{\partial m_2}"
            r"\left[ -\ln\left(\frac{e^{m_2x+b_2}}{e^{m_1x+b_1} + e^{m_2x+b_2} + e^{m_3x+b_3}}\right) \right]"
        )
        final_eq[21:27].set_color(COOL_YELLOW)
        final_eq[29:35].set_color(COOL_BLUE)
        final_eq[37:43].set_color(COOL_YELLOW)
        final_eq[45:51].set_color(COOL_GREEN)
        
        h_eqs = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).move_to(([-5.13787109, 0, 0]))
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN)
        
        center = [(h_eqs.get_right()[0] + RIGHT_SIDE[0]) / 2, 0, 0]
        
        center_eqs.move_to(center)
        
        center_eqs_2 = VGroup(softmax_eq_2, cross_entropy_loss_eq_2).arrange(DOWN).move_to(center)
        big_eq.move_to(center)
        big_eq_2.move_to(center)
        big_eq_3.move_to(center)
        big_eq_4.move_to(ORIGIN)
        
            
        self.play(Write(center_eqs), Write(h_eqs))
        
        self.wait()
        
        self.play(
            ReplacementTransform(center_eqs, center_eqs_2)
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(cross_entropy_loss_eq_2[0:9], big_eq[0:9]),
            ReplacementTransform(cross_entropy_loss_eq_2[12], big_eq[24]),
            ReplacementTransform(softmax_eq_2[4:19], big_eq[9:24]),
            FadeOut(VGroup(cross_entropy_loss_eq_2[9:12], softmax_eq_2[0:4]))
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq[0:9], big_eq_2[0:9]),
            ReplacementTransform(big_eq[24], big_eq_2[28]),
            ReplacementTransform(big_eq[9:14], big_eq_2[9:14]),
            ReplacementTransform(big_eq[15:24], big_eq_2[19:28]),
            ReplacementTransform(h1_eq[3:9], big_eq_2[14:20]),
            FadeOut(h1_eq[0:3]),
            FadeOut(big_eq[14:16]),
        )
            
        self.wait()    
                                
        self.play(
            ReplacementTransform(big_eq_2[0:10], big_eq_3[0:10]),
            ReplacementTransform(big_eq_2[28], big_eq_3[36]),
            ReplacementTransform(big_eq_2[12], big_eq_3[16]),
            ReplacementTransform(big_eq_2[13:22], big_eq_3[17:26]),
            ReplacementTransform(big_eq_2[24:28], big_eq_3[32:36]),
            FadeOut(VGroup(big_eq_2[10:12], big_eq_2[22:24], h2_eq[0:3])),
            ReplacementTransform(h2_eq[3:9].copy(), big_eq_3[10:16]),
            ReplacementTransform(h2_eq[3:9], big_eq_3[26:32]),
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq_3[0:35], big_eq_4[0:35]),
            ReplacementTransform(big_eq_3[36], big_eq_4[40]),
            FadeOut(big_eq_3[34:36]),
            ReplacementTransform(h3_eq[3:9], big_eq_4[34:40]),
            FadeOut(h3_eq[0:3])
        )
        
        self.wait()
        
        self.play(big_eq_4.animate.shift(UP * 2))
        
        self.wait()
        
        self.play(
            Write(final_eq[0]),
            Write(final_eq[5:16]),
            ReplacementTransform(big_eq_4[0:4], final_eq[1:5]),
            ReplacementTransform(big_eq_4[4], final_eq[9]),
            Write(final_eq[52]),
            ReplacementTransform(big_eq_4[5:41].copy(), final_eq[16:52])
        )      
        
class P27V3Slow(InteractiveScene):
    def construct(self):
        h1_eq = Tex(r"h_1 = m_1 x + b_1").set_color(COOL_BLUE)
        h2_eq = Tex(r"h_2 = m_2 x + b_2").set_color(COOL_YELLOW)
        h3_eq = Tex(r"h_3 = m_3 x + b_3").set_color(COOL_GREEN)
        
        softmax_eq = Tex(r"\hat{y}_i = \frac{e^{h_i}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq[9:11].set_color(COOL_BLUE)
        softmax_eq[13:15].set_color(COOL_YELLOW)
        softmax_eq[17:19].set_color(COOL_GREEN)
        
        cross_entropy_loss_eq = Tex(r"Loss = -\ln(\hat{y}_i)")
        softmax_eq_2 = Tex(r"\hat{y}_2 = \frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}}")
        softmax_eq_2[5:7].set_color(COOL_YELLOW)
        softmax_eq_2[9:11].set_color(COOL_BLUE)
        softmax_eq_2[13:15].set_color(COOL_YELLOW)
        softmax_eq_2[17:19].set_color(COOL_GREEN)
        
        cross_entropy_loss_eq_2 = Tex(r"Loss = -\ln(\hat{y}_2)")
        
        big_eq = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{h_1} + e^{h_2} + e^{h_3}})")
        big_eq[10:12].set_color(COOL_YELLOW)
        big_eq[14:16].set_color(COOL_BLUE)
        big_eq[18:20].set_color(COOL_YELLOW)
        big_eq[22:24].set_color(COOL_GREEN)
        
        big_eq_2 = Tex(r"Loss = -\ln(\frac{e^{h_2}}{e^{m_1 x + b_1} + e^{h_2} + e^{h_3}})")
        big_eq_2[10:12].set_color(COOL_YELLOW)
        big_eq_2[14:20].set_color(COOL_BLUE)
        big_eq_2[22:24].set_color(COOL_YELLOW)
        big_eq_2[26:28].set_color(COOL_GREEN)
        
        
        big_eq_3 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{h_3}})")
        big_eq_3[10:16].set_color(COOL_YELLOW)
        big_eq_3[18:24].set_color(COOL_BLUE)
        big_eq_3[26:32].set_color(COOL_YELLOW)
        big_eq_3[34:36].set_color(COOL_GREEN)
        
        big_eq_4 = Tex(r"Loss = -\ln(\frac{e^{m_2 x + b_2}}{e^{m_1 x + b_1} + e^{m_2 x + b_2} + e^{m_3 x + b_3}})")
        big_eq_4[10:16].set_color(COOL_YELLOW)
        big_eq_4[18:24].set_color(COOL_BLUE)
        big_eq_4[26:32].set_color(COOL_YELLOW)
        big_eq_4[34:40].set_color(COOL_GREEN)
        
        final_eq = Tex(
            r"\frac{\partial Loss}{\partial m_2} = \frac{\partial}{\partial m_2}"
            r"\left[ -\ln\left(\frac{e^{m_2x+b_2}}{e^{m_1x+b_1} + e^{m_2x+b_2} + e^{m_3x+b_3}}\right) \right]"
        )
        final_eq[21:27].set_color(COOL_YELLOW)
        final_eq[29:35].set_color(COOL_BLUE)
        final_eq[37:43].set_color(COOL_YELLOW)
        final_eq[45:51].set_color(COOL_GREEN)
        
        h_eqs = VGroup(h1_eq, h2_eq, h3_eq).arrange(DOWN).move_to(([-5.13787109, 0, 0]))
        center_eqs = VGroup(softmax_eq, cross_entropy_loss_eq).arrange(DOWN)
        
        center = [(h_eqs.get_right()[0] + RIGHT_SIDE[0]) / 2, 0, 0]
        
        center_eqs.move_to(center)
        
        center_eqs_2 = VGroup(softmax_eq_2, cross_entropy_loss_eq_2).arrange(DOWN).move_to(center)
        big_eq.move_to(center)
        big_eq_2.move_to(center)
        big_eq_3.move_to(center)
        big_eq_4.move_to(ORIGIN)
        
            
        self.play(Write(center_eqs), Write(h_eqs), run_time=2)
        
        self.wait()
        
        self.play(
            ReplacementTransform(center_eqs, center_eqs_2), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(cross_entropy_loss_eq_2[0:9], big_eq[0:9]),
            ReplacementTransform(cross_entropy_loss_eq_2[12], big_eq[24]),
            ReplacementTransform(softmax_eq_2[4:19], big_eq[9:24]),
            FadeOut(VGroup(cross_entropy_loss_eq_2[9:12], softmax_eq_2[0:4])), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq[0:9], big_eq_2[0:9]),
            ReplacementTransform(big_eq[24], big_eq_2[28]),
            ReplacementTransform(big_eq[9:14], big_eq_2[9:14]),
            ReplacementTransform(big_eq[15:24], big_eq_2[19:28]),
            ReplacementTransform(h1_eq[3:9], big_eq_2[14:20]),
            FadeOut(h1_eq[0:3]),
            FadeOut(big_eq[14:16]), run_time=2
        )
            
        self.wait()    
                                
        self.play(
            ReplacementTransform(big_eq_2[0:10], big_eq_3[0:10]),
            ReplacementTransform(big_eq_2[28], big_eq_3[36]),
            ReplacementTransform(big_eq_2[12], big_eq_3[16]),
            ReplacementTransform(big_eq_2[13:22], big_eq_3[17:26]),
            ReplacementTransform(big_eq_2[24:28], big_eq_3[32:36]),
            FadeOut(VGroup(big_eq_2[10:12], big_eq_2[22:24], h2_eq[0:3])),
            ReplacementTransform(h2_eq[3:9].copy(), big_eq_3[10:16]),
            ReplacementTransform(h2_eq[3:9], big_eq_3[26:32]), run_time=2
        )
        
        self.wait()
        
        self.play(
            ReplacementTransform(big_eq_3[0:35], big_eq_4[0:35]),
            ReplacementTransform(big_eq_3[36], big_eq_4[40]),
            FadeOut(big_eq_3[34:36]),
            ReplacementTransform(h3_eq[3:9], big_eq_4[34:40]),
            FadeOut(h3_eq[0:3]), run_time=2
        )
        
        self.wait()
        
        self.play(big_eq_4.animate.shift(UP * 2), run_time=2)
        
        self.wait()
        
        self.play(
            Write(final_eq[0]),
            Write(final_eq[5:16]),
            ReplacementTransform(big_eq_4[0:4], final_eq[1:5]),
            ReplacementTransform(big_eq_4[4], final_eq[9]),
            Write(final_eq[52]),
            ReplacementTransform(big_eq_4[5:41].copy(), final_eq[16:52]), run_time=2
        )
        
