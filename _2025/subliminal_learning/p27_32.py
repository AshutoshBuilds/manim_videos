from manimlib import *
from MF_Tools import *

CHILL_BROWN='#958a7a'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

class P27(Scene):
    def construct(self):
        p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:]
        p27_to_manim_2 = SVGMobject("p27_to_manim-02.svg")[1:].scale(4)
        p27_to_manim_3 = SVGMobject("p27_to_manim-03.svg")[1:]
        p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:]
        p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:]
        p27_to_manim_6 = SVGMobject("p27_to_manim-06.svg")[1:]
        p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:]
                
        n2_1 = p27_to_manim_2.submobjects[10]
        n3_2 = p27_to_manim_2.submobjects[9]
        n1_1 = p27_to_manim_2.submobjects[11]
        n1_2 = p27_to_manim_2.submobjects[12]
        n2_2 = p27_to_manim_2.submobjects[13]
        n3_1 = p27_to_manim_2.submobjects[8]
        
        n3_1_arrow = p27_to_manim_2.submobjects[14]
        n3_2_arrow = p27_to_manim_2.submobjects[15]
        
        n2_1_graph = p27_to_manim_2.submobjects[16]
        n2_2_graph = p27_to_manim_2.submobjects[21]
        
        n2_1_text = VGroup(p27_to_manim_2.submobjects[17], p27_to_manim_2.submobjects[18], p27_to_manim_2.submobjects[19], p27_to_manim_2.submobjects[20])
        n2_2_text = VGroup(p27_to_manim_2.submobjects[22], p27_to_manim_2.submobjects[23], p27_to_manim_2.submobjects[24], p27_to_manim_2.submobjects[25])

        left_dimension = VGroup(p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]).scale(4).next_to(VGroup(n1_1, n1_2), LEFT)
        top_dimension = VGroup(p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]).scale(4).next_to(VGroup(n2_1, n3_1), UP)

        x1 = Tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = Tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = Tex(
            r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(n2_2, DOWN)
        
        model_parameters_text = Text('Model Parameters*', font='Myriad Pro', font_size=24).next_to(model_parameters, DOWN).set_color(CHILL_BROWN)

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = normalize(c2_center - c1_center)

            ul, ur, dl, dr = c1.get_critical_point(UL), c1.get_critical_point(UR), c1.get_critical_point(DL), c1.get_critical_point(DR)

            width = np.linalg.norm(ur - ul)
            height = np.linalg.norm(ul - dl)
            radius = max(width, height) / 2

            return c1_center + radius * direction
        
        def split_line(line, break_width, split_ratio):
            start = line.get_start()
            end = line.get_end()
            direction = (end - start) / line.get_length()

            split_point = start + direction * line.get_length() * split_ratio

            half_gap = direction * (break_width / 2)

            line1 = Line(start, split_point - half_gap)
            line2 = Line(split_point + half_gap, end)

            return line1, line2, split_point
        
        def animate_split_line(line, split_line_1, split_line_2, split_ratio):
            line_part_1, line_part_2, _ = split_line(line, 0, split_ratio)
            line_part_1.set_color(split_line_1.get_color())
            line_part_2.set_color(split_line_2.get_color())
            self.remove(line)
            return AnimationGroup(
                ReplacementTransform(line_part_1, split_line_1),
                ReplacementTransform(line_part_2, split_line_2)
            )



        connections = VGroup(
            Line(edge_point(n1_1, n2_1), edge_point(n2_1, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_1, n2_2), edge_point(n2_2, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_1), edge_point(n2_1, n1_2), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_2), edge_point(n2_2, n1_2), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_1), edge_point(n3_1, n2_1), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_2), edge_point(n3_2, n2_1), color=CHILL_BROWN),
            Line(edge_point(n2_2, n3_1), edge_point(n3_1, n2_2), color=FRESH_TAN),
            Line(edge_point(n2_2, n3_2), edge_point(n3_2, n2_2), color=CHILL_BROWN),
        )
        
        ln1_1_n2_1 = connections[0]
        ln1_1_n2_2 = connections[1]
        ln1_2_n2_1 = connections[2]
        ln1_2_n2_2 = connections[3]
        ln2_1_n3_1 = connections[4]
        ln2_1_n3_2 = connections[5]
        ln2_2_n3_1 = connections[6]
        ln2_2_n3_2 = connections[7]
        
        theta_1 = Tex(r"\theta_1").scale(0.6).set_color(FRESH_TAN)
        theta_2 = Tex(r"\theta_2").scale(0.6).set_color(FRESH_TAN)
        theta_3 = Tex(r"\theta_3").scale(0.6).set_color(FRESH_TAN)
        theta_4 = Tex(r"\theta_4").scale(0.6).set_color(FRESH_TAN)
        theta_5 = Tex(r"\theta_5").scale(0.6).set_color(FRESH_TAN)
        theta_6 = Tex(r"\theta_6").scale(0.6).set_color(FRESH_TAN)
        theta_7 = Tex(r"\theta_7").scale(0.6).set_color(CHILL_BROWN)
        theta_8 = Tex(r"\theta_8").scale(0.6).set_color(CHILL_BROWN)

        
        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(ln1_1_n2_1, theta_1.get_width() + 0.2, 0.5)
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(ln1_2_n2_1, theta_2.get_width() + 0.2, 0.75)
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(ln1_1_n2_2, theta_3.get_width() + 0.2, 0.75)
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(ln1_2_n2_2, theta_4.get_width() + 0.2, 0.5)
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(ln2_1_n3_1, theta_5.get_width() + 0.2, 0.5)
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(ln2_2_n3_1, theta_6.get_width() + 0.2, 0.75)
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(ln2_1_n3_2, theta_7.get_width() + 0.2, 0.75)
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(ln2_2_n3_2, theta_8.get_width() + 0.2, 0.5)
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = AnimationGroup(
            FadeIn(VGroup(n1_1, n1_2, x1, x2))
        )

        group2 = AnimationGroup(
            GrowArrow(ln1_1_n2_1),
            GrowArrow(ln1_1_n2_2),
            GrowArrow(ln1_2_n2_1),
            GrowArrow(ln1_2_n2_2)
        )

        group3 = AnimationGroup(
            FadeIn(VGroup(n2_1, n2_2))
        )

        group4 = AnimationGroup(
            FadeIn(VGroup(n2_1_graph, n2_2_graph)),
            Write(n2_1_text),
            Write(n2_2_text),
        )

        group5 = AnimationGroup(
            GrowArrow(ln2_1_n3_1),
            GrowArrow(ln2_1_n3_2),
            GrowArrow(ln2_2_n3_1),
            GrowArrow(ln2_2_n3_2),
        )

        group6 = AnimationGroup(
            FadeIn(VGroup(n3_1, n3_2)),
            GrowFromEdge(n3_1_arrow, LEFT),
            GrowFromEdge(n3_2_arrow, LEFT)
        )

        self.play(
            LaggedStart(
                group1, group2, group3, group4, group5, group6,
                lag_ratio=0.2,
                run_time=2.5
            )
        )
        
        self.wait(1)
        
        self.add(left_dimension, top_dimension)
        
        self.wait(1)

        theta1_group = AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            Write(theta_1),
            lag_ratio=0.5
        )

        theta2_group = AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            Write(theta_2),
            lag_ratio=0.5
        )

        theta3_group = AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            Write(theta_3),
            lag_ratio=0.5
        )

        theta4_group = AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            Write(theta_4),
            lag_ratio=0.5
        )

        theta5_group = AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            Write(theta_5),
            lag_ratio=0.5
        )

        theta6_group = AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            Write(theta_6),
            lag_ratio=0.5
        )

        theta7_group = AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            Write(theta_7),
            lag_ratio=0.5
        )

        theta8_group = AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            Write(theta_8),
            lag_ratio=0.5
        )

        self.play(
            LaggedStart(
                theta1_group,
                theta2_group,
                theta3_group,
                theta4_group,
                theta5_group,
                theta6_group,
                theta7_group,
                theta8_group,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        model_parameters_1 = AnimationGroup(
            FadeIn(model_parameters[0:3])
        )
        
        model_parameters_2 = AnimationGroup(
            ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            FadeIn(model_parameters[5])
        )
        
        model_parameters_3 = AnimationGroup(
            ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            FadeIn(model_parameters[8])
        )
        
        model_parameters_4 = AnimationGroup(
            ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            FadeIn(model_parameters[11])
        )

        model_parameters_5 = AnimationGroup(
            ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            FadeIn(model_parameters[14])
        )

        model_parameters_6 = AnimationGroup(
            ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            FadeIn(model_parameters[17])
        )

        model_parameters_7 = AnimationGroup(
            ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            FadeIn(model_parameters[20])
        )

        model_parameters_8 = AnimationGroup(
            ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            FadeIn(model_parameters[23])
        )

        model_parameters_9 = AnimationGroup(
            ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            FadeIn(model_parameters[26])
        )

        self.play(
            LaggedStart(
                model_parameters_1,
                model_parameters_2,
                model_parameters_3,
                model_parameters_4,
                model_parameters_5,
                model_parameters_6,
                model_parameters_7,
                model_parameters_8,
                model_parameters_9,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        self.play(FadeIn(model_parameters_text))
        
        self.wait(1)

        self.embed()
        
class P27_28(Scene):
    def construct(self):
        # p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:].scale(4)
        p27_to_manim_2 = SVGMobject("p27_to_manim-02.svg")[1:].scale(4)
        p27_to_manim_3 = SVGMobject("p27_to_manim-03.svg")[1:].scale(4)
        # p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:].scale(4)
        # p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:].scale(4)
        p27_to_manim_6 = SVGMobject("p27_to_manim-06.svg")[1:].scale(4)
        # p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:].scale(4)
        p28_to_manim_4 = SVGMobject("p28_to_manim-04.svg")[1:].scale(4)
        

        mnist_network = ImageMobject("mnist_network.png")
        

        n2_1 = p27_to_manim_2.submobjects[10]
        n3_2 = p27_to_manim_2.submobjects[9]
        n1_1 = p27_to_manim_2.submobjects[11]
        n1_2 = p27_to_manim_2.submobjects[12]
        n2_2 = p27_to_manim_2.submobjects[13]
        n3_1 = p27_to_manim_2.submobjects[8]
        
        n3_1_arrow = p27_to_manim_2.submobjects[14]
        n3_2_arrow = p27_to_manim_2.submobjects[15]
        
        n2_1_graph = p27_to_manim_2.submobjects[16]
        n2_2_graph = p27_to_manim_2.submobjects[21]
        
        n2_1_text = VGroup(p27_to_manim_2.submobjects[17], p27_to_manim_2.submobjects[18], p27_to_manim_2.submobjects[19], p27_to_manim_2.submobjects[20])
        n2_2_text = VGroup(p27_to_manim_2.submobjects[22], p27_to_manim_2.submobjects[23], p27_to_manim_2.submobjects[24], p27_to_manim_2.submobjects[25])

        left_dimension = VGroup(p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]).next_to(VGroup(n1_1, n1_2), LEFT)
        top_dimension = VGroup(p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]).next_to(VGroup(n2_1, n3_1), UP)

        x1 = Tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = Tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = Tex(
            r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(n2_2, DOWN)
        
        model_parameters_text = Text('Model Parameters*', font='Myriad Pro', font_size=24).next_to(model_parameters, DOWN).set_color(CHILL_BROWN)

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = normalize(c2_center - c1_center)

            ul, ur, dl, dr = c1.get_critical_point(UL), c1.get_critical_point(UR), c1.get_critical_point(DL), c1.get_critical_point(DR)

            width = np.linalg.norm(ur - ul)
            height = np.linalg.norm(ul - dl)
            radius = max(width, height) / 2

            return c1_center + radius * direction
        
        def split_line(line, break_width, split_ratio):
            start = line.get_start()
            end = line.get_end()
            direction = (end - start) / line.get_length()

            split_point = start + direction * line.get_length() * split_ratio

            half_gap = direction * (break_width / 2)

            line1 = Line(start, split_point - half_gap)
            line2 = Line(split_point + half_gap, end)

            return line1, line2, split_point
        
        def animate_split_line(line, split_line_1, split_line_2, split_ratio):
            line_part_1, line_part_2, _ = split_line(line, 0, split_ratio)
            line_part_1.set_color(split_line_1.get_color())
            line_part_2.set_color(split_line_2.get_color())
            self.remove(line)
            return AnimationGroup(
                ReplacementTransform(line_part_1, split_line_1),
                ReplacementTransform(line_part_2, split_line_2)
            )



        connections = VGroup(
            Line(edge_point(n1_1, n2_1), edge_point(n2_1, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_1, n2_2), edge_point(n2_2, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_1), edge_point(n2_1, n1_2), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_2), edge_point(n2_2, n1_2), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_1), edge_point(n3_1, n2_1), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_2), edge_point(n3_2, n2_1), color=CHILL_BROWN),
            Line(edge_point(n2_2, n3_1), edge_point(n3_1, n2_2), color=FRESH_TAN),
            Line(edge_point(n2_2, n3_2), edge_point(n3_2, n2_2), color=CHILL_BROWN),
        )
        
        ln1_1_n2_1 = connections[0]
        ln1_1_n2_2 = connections[1]
        ln1_2_n2_1 = connections[2]
        ln1_2_n2_2 = connections[3]
        ln2_1_n3_1 = connections[4]
        ln2_1_n3_2 = connections[5]
        ln2_2_n3_1 = connections[6]
        ln2_2_n3_2 = connections[7]
        
        theta_1 = Tex(r"\theta_1").scale(0.6).set_color(FRESH_TAN)
        theta_2 = Tex(r"\theta_2").scale(0.6).set_color(FRESH_TAN)
        theta_3 = Tex(r"\theta_3").scale(0.6).set_color(FRESH_TAN)
        theta_4 = Tex(r"\theta_4").scale(0.6).set_color(FRESH_TAN)
        theta_5 = Tex(r"\theta_5").scale(0.6).set_color(FRESH_TAN)
        theta_6 = Tex(r"\theta_6").scale(0.6).set_color(FRESH_TAN)
        theta_7 = Tex(r"\theta_7").scale(0.6).set_color(CHILL_BROWN)
        theta_8 = Tex(r"\theta_8").scale(0.6).set_color(CHILL_BROWN)

        
        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(ln1_1_n2_1, theta_1.get_width() + 0.2, 0.5)
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(ln1_2_n2_1, theta_2.get_width() + 0.2, 0.75)
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(ln1_1_n2_2, theta_3.get_width() + 0.2, 0.75)
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(ln1_2_n2_2, theta_4.get_width() + 0.2, 0.5)
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(ln2_1_n3_1, theta_5.get_width() + 0.2, 0.5)
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(ln2_2_n3_1, theta_6.get_width() + 0.2, 0.75)
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(ln2_1_n3_2, theta_7.get_width() + 0.2, 0.75)
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(ln2_2_n3_2, theta_8.get_width() + 0.2, 0.5)
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = AnimationGroup(
            FadeIn(VGroup(n1_1, n1_2, x1, x2))
        )

        group2 = AnimationGroup(
            GrowArrow(ln1_1_n2_1),
            GrowArrow(ln1_1_n2_2),
            GrowArrow(ln1_2_n2_1),
            GrowArrow(ln1_2_n2_2)
        )

        group3 = AnimationGroup(
            FadeIn(VGroup(n2_1, n2_2))
        )

        group4 = AnimationGroup(
            FadeIn(VGroup(n2_1_graph, n2_2_graph)),
            Write(n2_1_text),
            Write(n2_2_text),
        )

        group5 = AnimationGroup(
            GrowArrow(ln2_1_n3_1),
            GrowArrow(ln2_1_n3_2),
            GrowArrow(ln2_2_n3_1),
            GrowArrow(ln2_2_n3_2),
        )

        group6 = AnimationGroup(
            FadeIn(VGroup(n3_1, n3_2)),
            GrowFromEdge(n3_1_arrow, LEFT),
            GrowFromEdge(n3_2_arrow, LEFT)
        )

        self.play(
            LaggedStart(
                group1, group2, group3, group4, group5, group6,
                lag_ratio=0.2,
                run_time=2.5
            )
        )
        
        self.wait(1)
        
        self.add(left_dimension, top_dimension)
        
        self.wait(1)

        theta1_group = AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            Write(theta_1),
            lag_ratio=0.5
        )

        theta2_group = AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            Write(theta_2),
            lag_ratio=0.5
        )

        theta3_group = AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            Write(theta_3),
            lag_ratio=0.5
        )

        theta4_group = AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            Write(theta_4),
            lag_ratio=0.5
        )

        theta5_group = AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            Write(theta_5),
            lag_ratio=0.5
        )

        theta6_group = AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            Write(theta_6),
            lag_ratio=0.5
        )

        theta7_group = AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            Write(theta_7),
            lag_ratio=0.5
        )

        theta8_group = AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            Write(theta_8),
            lag_ratio=0.5
        )

        self.play(
            LaggedStart(
                theta1_group,
                theta2_group,
                theta3_group,
                theta4_group,
                theta5_group,
                theta6_group,
                theta7_group,
                theta8_group,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        model_parameters_1 = AnimationGroup(
            FadeIn(model_parameters[0:3])
        )
        
        model_parameters_2 = AnimationGroup(
            ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            FadeIn(model_parameters[5])
        )
        
        model_parameters_3 = AnimationGroup(
            ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            FadeIn(model_parameters[8])
        )
        
        model_parameters_4 = AnimationGroup(
            ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            FadeIn(model_parameters[11])
        )

        model_parameters_5 = AnimationGroup(
            ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            FadeIn(model_parameters[14])
        )

        model_parameters_6 = AnimationGroup(
            ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            FadeIn(model_parameters[17])
        )

        model_parameters_7 = AnimationGroup(
            ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            FadeIn(model_parameters[20])
        )

        model_parameters_8 = AnimationGroup(
            ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            FadeIn(model_parameters[23])
        )

        model_parameters_9 = AnimationGroup(
            ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            FadeIn(model_parameters[26])
        )

        self.play(
            LaggedStart(
                model_parameters_1,
                model_parameters_2,
                model_parameters_3,
                model_parameters_4,
                model_parameters_5,
                model_parameters_6,
                model_parameters_7,
                model_parameters_8,
                model_parameters_9,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        self.camera.frame.save_state()
        
        
        self.play(FadeIn(model_parameters_text))
        
        self.wait(1)
        
        self.play(self.camera.frame.animate.move_to([3.5, 0, 0]))
        
        self.wait(1)
        
        p27_to_manim_6[29].next_to(n3_1_arrow, RIGHT, buff=0.1)
        p27_to_manim_6[28].next_to(n3_2_arrow, RIGHT, buff=0.1)

        p27_to_manim_6[0:13].next_to(p27_to_manim_6[29], DOWN, buff=0.1)
        p27_to_manim_6[13:28].next_to(p27_to_manim_6[28], DOWN, buff=0.1)

        primary_output = VGroup(p27_to_manim_6[29], p27_to_manim_6[0:13])
        auxiliary_output = VGroup(p27_to_manim_6[28], p27_to_manim_6[13:28])
        
        outputs = VGroup(primary_output, auxiliary_output)
        
        self.play(FadeIn(primary_output), FadeIn(auxiliary_output))
        
        self.wait(1)
        
        # p28_to_manim_4.move_to([(primary_output.get_x() + p28_to_manim_4.get_width()/2) - (p28_to_manim_4.submobjects[1].get_width()/2), 0, 0])

        p28_to_manim_4.move_to([
            (primary_output.get_x() + p28_to_manim_4.get_width()/2) - (p28_to_manim_4.submobjects[1].get_width()/2),
            0,
            0
        ])

        top_outputs = outputs.get_top()[1]
        bottom_outputs = outputs.get_bottom()[1]
        bottom_submob1 = p28_to_manim_4.submobjects[1].get_bottom()[1]
        top_submob5 = p28_to_manim_4.submobjects[5].get_top()[1]
        vertical_shift = (top_outputs - bottom_submob1 - top_submob5 + bottom_outputs) / 2

        # Apply the shift
        p28_to_manim_4.shift([0, vertical_shift, 0])
        
        # MATH TIME ;(
        '''
        p28_to_manim_4.get_width() = 5.860444245515045
        was about to pull out the ti84 for this lol
        '''

        self.play(FadeIn(p28_to_manim_4))
        
        self.wait(1)
        
        mnist_network.move_to([7.325, -0.175,  0])
        
        self.bring_to_back(mnist_network)
        #when mnist gets updated
        # self.play(FadeIn(mnist_network))
        
        self.wait(1)
        

        self.embed()
        
class P27_29(Scene):
    def construct(self):
        # p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:].scale(4)
        p27_to_manim_2 = SVGMobject("p27_to_manim-02.svg")[1:].scale(4)
        p27_to_manim_3 = SVGMobject("p27_to_manim-03.svg")[1:].scale(4)
        # p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:].scale(4)
        # p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:].scale(4)
        p27_to_manim_6 = SVGMobject("p27_to_manim-06.svg")[1:].scale(4)
        # p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:].scale(4)
        p28_to_manim_4 = SVGMobject("p28_to_manim-04.svg")[1:].scale(4)
        p29_to_32_to_manim_1 = SVGMobject("p29_to_32_to_manim-01.svg")[1:].scale(4)
        

        mnist_network = ImageMobject("mnist_network.png")
        

        n2_1 = p27_to_manim_2.submobjects[10]
        n3_2 = p27_to_manim_2.submobjects[9]
        n1_1 = p27_to_manim_2.submobjects[11]
        n1_2 = p27_to_manim_2.submobjects[12]
        n2_2 = p27_to_manim_2.submobjects[13]
        n3_1 = p27_to_manim_2.submobjects[8]
        
        n3_1_arrow = p27_to_manim_2.submobjects[14]
        n3_2_arrow = p27_to_manim_2.submobjects[15]
        
        n2_1_graph = p27_to_manim_2.submobjects[16]
        n2_2_graph = p27_to_manim_2.submobjects[21]
        
        n2_1_text = VGroup(p27_to_manim_2.submobjects[17], p27_to_manim_2.submobjects[18], p27_to_manim_2.submobjects[19], p27_to_manim_2.submobjects[20])
        n2_2_text = VGroup(p27_to_manim_2.submobjects[22], p27_to_manim_2.submobjects[23], p27_to_manim_2.submobjects[24], p27_to_manim_2.submobjects[25])

        left_dimension = VGroup(p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]).next_to(VGroup(n1_1, n1_2), LEFT)
        top_dimension = VGroup(p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]).next_to(VGroup(n2_1, n3_1), UP)

        x1 = Tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = Tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = Tex(
            r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(n2_2, DOWN)
        
        model_parameters_text = Text('Model Parameters*', font='Myriad Pro', font_size=24).next_to(model_parameters, DOWN).set_color(CHILL_BROWN)

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = normalize(c2_center - c1_center)

            ul, ur, dl, dr = c1.get_critical_point(UL), c1.get_critical_point(UR), c1.get_critical_point(DL), c1.get_critical_point(DR)

            width = np.linalg.norm(ur - ul)
            height = np.linalg.norm(ul - dl)
            radius = max(width, height) / 2

            return c1_center + radius * direction
        
        def split_line(line, break_width, split_ratio):
            start = line.get_start()
            end = line.get_end()
            direction = (end - start) / line.get_length()

            split_point = start + direction * line.get_length() * split_ratio

            half_gap = direction * (break_width / 2)

            line1 = Line(start, split_point - half_gap)
            line2 = Line(split_point + half_gap, end)

            return line1, line2, split_point
        
        def animate_split_line(line, split_line_1, split_line_2, split_ratio):
            line_part_1, line_part_2, _ = split_line(line, 0, split_ratio)
            line_part_1.set_color(split_line_1.get_color())
            line_part_2.set_color(split_line_2.get_color())
            self.remove(line)
            return AnimationGroup(
                ReplacementTransform(line_part_1, split_line_1),
                ReplacementTransform(line_part_2, split_line_2)
            )



        connections = VGroup(
            Line(edge_point(n1_1, n2_1), edge_point(n2_1, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_1, n2_2), edge_point(n2_2, n1_1), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_1), edge_point(n2_1, n1_2), color=FRESH_TAN),
            Line(edge_point(n1_2, n2_2), edge_point(n2_2, n1_2), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_1), edge_point(n3_1, n2_1), color=FRESH_TAN),
            Line(edge_point(n2_1, n3_2), edge_point(n3_2, n2_1), color=CHILL_BROWN),
            Line(edge_point(n2_2, n3_1), edge_point(n3_1, n2_2), color=FRESH_TAN),
            Line(edge_point(n2_2, n3_2), edge_point(n3_2, n2_2), color=CHILL_BROWN),
        )
        
        ln1_1_n2_1 = connections[0]
        ln1_1_n2_2 = connections[1]
        ln1_2_n2_1 = connections[2]
        ln1_2_n2_2 = connections[3]
        ln2_1_n3_1 = connections[4]
        ln2_1_n3_2 = connections[5]
        ln2_2_n3_1 = connections[6]
        ln2_2_n3_2 = connections[7]
        
        theta_1 = Tex(r"\theta_1").scale(0.6).set_color(FRESH_TAN)
        theta_2 = Tex(r"\theta_2").scale(0.6).set_color(FRESH_TAN)
        theta_3 = Tex(r"\theta_3").scale(0.6).set_color(FRESH_TAN)
        theta_4 = Tex(r"\theta_4").scale(0.6).set_color(FRESH_TAN)
        theta_5 = Tex(r"\theta_5").scale(0.6).set_color(FRESH_TAN)
        theta_6 = Tex(r"\theta_6").scale(0.6).set_color(FRESH_TAN)
        theta_7 = Tex(r"\theta_7").scale(0.6).set_color(CHILL_BROWN)
        theta_8 = Tex(r"\theta_8").scale(0.6).set_color(CHILL_BROWN)

        
        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(ln1_1_n2_1, theta_1.get_width() + 0.2, 0.5)
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(ln1_2_n2_1, theta_2.get_width() + 0.2, 0.75)
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(ln1_1_n2_2, theta_3.get_width() + 0.2, 0.75)
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(ln1_2_n2_2, theta_4.get_width() + 0.2, 0.5)
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(ln2_1_n3_1, theta_5.get_width() + 0.2, 0.5)
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(ln2_2_n3_1, theta_6.get_width() + 0.2, 0.75)
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(ln2_1_n3_2, theta_7.get_width() + 0.2, 0.75)
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(ln2_2_n3_2, theta_8.get_width() + 0.2, 0.5)
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = AnimationGroup(
            FadeIn(VGroup(n1_1, n1_2, x1, x2))
        )

        group2 = AnimationGroup(
            GrowArrow(ln1_1_n2_1),
            GrowArrow(ln1_1_n2_2),
            GrowArrow(ln1_2_n2_1),
            GrowArrow(ln1_2_n2_2)
        )

        group3 = AnimationGroup(
            FadeIn(VGroup(n2_1, n2_2))
        )

        group4 = AnimationGroup(
            FadeIn(VGroup(n2_1_graph, n2_2_graph)),
            Write(n2_1_text),
            Write(n2_2_text),
        )

        group5 = AnimationGroup(
            GrowArrow(ln2_1_n3_1),
            GrowArrow(ln2_1_n3_2),
            GrowArrow(ln2_2_n3_1),
            GrowArrow(ln2_2_n3_2),
        )

        group6 = AnimationGroup(
            FadeIn(VGroup(n3_1, n3_2)),
            GrowFromEdge(n3_1_arrow, LEFT),
            GrowFromEdge(n3_2_arrow, LEFT)
        )

        self.play(
            LaggedStart(
                group1, group2, group3, group4, group5, group6,
                lag_ratio=0.2,
                run_time=2.5
            )
        )
        
        self.wait(1)
        
        self.add(left_dimension, top_dimension)
        
        self.wait(1)

        theta1_group = AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            Write(theta_1),
            lag_ratio=0.5
        )

        theta2_group = AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            Write(theta_2),
            lag_ratio=0.5
        )

        theta3_group = AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            Write(theta_3),
            lag_ratio=0.5
        )

        theta4_group = AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            Write(theta_4),
            lag_ratio=0.5
        )

        theta5_group = AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            Write(theta_5),
            lag_ratio=0.5
        )

        theta6_group = AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            Write(theta_6),
            lag_ratio=0.5
        )

        theta7_group = AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            Write(theta_7),
            lag_ratio=0.5
        )

        theta8_group = AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            Write(theta_8),
            lag_ratio=0.5
        )

        self.play(
            LaggedStart(
                theta1_group,
                theta2_group,
                theta3_group,
                theta4_group,
                theta5_group,
                theta6_group,
                theta7_group,
                theta8_group,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        model_parameters_1 = AnimationGroup(
            FadeIn(model_parameters[0:3])
        )
        
        model_parameters_2 = AnimationGroup(
            ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            FadeIn(model_parameters[5])
        )
        
        model_parameters_3 = AnimationGroup(
            ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            FadeIn(model_parameters[8])
        )
        
        model_parameters_4 = AnimationGroup(
            ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            FadeIn(model_parameters[11])
        )

        model_parameters_5 = AnimationGroup(
            ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            FadeIn(model_parameters[14])
        )

        model_parameters_6 = AnimationGroup(
            ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            FadeIn(model_parameters[17])
        )

        model_parameters_7 = AnimationGroup(
            ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            FadeIn(model_parameters[20])
        )

        model_parameters_8 = AnimationGroup(
            ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            FadeIn(model_parameters[23])
        )

        model_parameters_9 = AnimationGroup(
            ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            FadeIn(model_parameters[26])
        )

        self.play(
            LaggedStart(
                model_parameters_1,
                model_parameters_2,
                model_parameters_3,
                model_parameters_4,
                model_parameters_5,
                model_parameters_6,
                model_parameters_7,
                model_parameters_8,
                model_parameters_9,
                lag_ratio=0.3,
                run_time=5
            )
        )
        
        self.wait(1)
        
        self.camera.frame.save_state()
        
        
        self.play(FadeIn(model_parameters_text))
        
        self.wait(1)
        
        self.play(self.camera.frame.animate.move_to([3, 0, 0]),
                  FadeOut(top_dimension),
                  FadeOut(left_dimension),
                  FadeOut(model_parameters_text))
        
        self.wait(1)
        
        p27_to_manim_6[29].next_to(n3_1_arrow, RIGHT, buff=0.1)
        p27_to_manim_6[28].next_to(n3_2_arrow, RIGHT, buff=0.1)

        p27_to_manim_6[0:13].next_to(p27_to_manim_6[29], DOWN, buff=0.1)
        p27_to_manim_6[13:28].next_to(p27_to_manim_6[28], DOWN, buff=0.1)

        primary_output = VGroup(p27_to_manim_6[29], p27_to_manim_6[0:13])
        auxiliary_output = VGroup(p27_to_manim_6[28], p27_to_manim_6[13:28])
        
        outputs = VGroup(primary_output, auxiliary_output)

        self.play(FadeIn(primary_output), FadeIn(auxiliary_output), model_parameters.animate.move_to([ 2.81398236e-01, -1.57497560e+00, -2.99721055e-17]))

        self.wait(1)

        # p28_to_manim_4.move_to([(primary_output.get_x() + p28_to_manim_4.get_width()/2) - (p28_to_manim_4.submobjects[1].get_width()/2), 0, 0])

        p28_to_manim_4.move_to([
            (primary_output.get_x() + p28_to_manim_4.get_width()/2) - (p28_to_manim_4.submobjects[1].get_width()/2),
            0,
            0
        ])

        top_outputs = outputs.get_top()[1]
        bottom_outputs = outputs.get_bottom()[1]
        bottom_submob1 = p28_to_manim_4.submobjects[1].get_bottom()[1]
        top_submob5 = p28_to_manim_4.submobjects[5].get_top()[1]
        vertical_shift = (top_outputs - bottom_submob1 - top_submob5 + bottom_outputs) / 2

        # Apply the shift
        p28_to_manim_4.shift([0, vertical_shift, 0])
        
        # MATH TIME ;(
        '''
        p28_to_manim_4.get_width() = 5.860444245515045
        was about to pull out the ti84 for this lol
        '''

        self.play(FadeIn(p28_to_manim_4))
        
        self.wait(1)
        
        mnist_network.move_to([7.325, -0.175,  0])
        
        self.bring_to_back(mnist_network)
        #when mnist gets updated
        # self.play(FadeIn(mnist_network))
        
        self.wait(1)
        
        # P29
        
        
        self.play(self.camera.frame.animate.move_to([0, 0, 0]),
                  FadeOut(mnist_network),
                  FadeOut(p28_to_manim_4)
                  )
        
        self.wait(1)
        
        t_model_parameters = Tex(
            r"\theta_T = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(VGroup(n1_1, auxiliary_output), DOWN)
        
        
        
        g_t = Tex(r"g_T").set_height(p27_to_manim_6[29].get_height()).move_to(p27_to_manim_6[29].get_center())
        f_t = Tex(r"f_T").set_height(p27_to_manim_6[28].get_height()).move_to(p27_to_manim_6[28].get_center())
        
        t_f_t = VGroup(p29_to_32_to_manim_1.submobjects[76], p29_to_32_to_manim_1.submobjects[183]).set_color(FRESH_TAN).move_to(primary_output[0].get_center())
        
        t_g_t = VGroup(p29_to_32_to_manim_1.submobjects[75], p29_to_32_to_manim_1.submobjects[184]).set_color(CHILL_BROWN).move_to(auxiliary_output[0].get_center())
        
        teacher_text = Text('TEACHER', font='Myriad Pro', font_size=48).next_to(VGroup(n1_1, primary_output, auxiliary_output, t_model_parameters), UP, buff=0.5).set_color(CHILL_BROWN)
        
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(model_parameters[1:27], t_model_parameters[2:28]),
                    ReplacementTransform(model_parameters[0], t_model_parameters[0]),
                    FadeIn(t_model_parameters[1]),
                    ReplacementTransform(primary_output[0], t_f_t[0]),
                    FadeIn(t_f_t[1]),
                    ReplacementTransform(auxiliary_output[0], t_g_t[0]),
                    FadeIn(t_g_t[1])
                ),
                Write(teacher_text),
                lag_ratio=0.8
            )
        )
        
        self.wait(1)

        
        
        s_auxiliary_output_text = auxiliary_output.copy()[1]
        s_primary_output_text = primary_output.copy()[1]
        
        t_network = VGroup(
            # Input neurons with labels
            n1_1, n1_2, x1, x2,
            
            # Hidden layer neurons and their activation functions
            n2_1, n2_2, n2_1_graph, n2_2_graph, n2_1_text, n2_2_text,
            
            # Output layer neurons and arrows
            n3_1, n3_2, n3_1_arrow, n3_2_arrow,
            
            # Split lines between input and hidden layer
            ln1_1_n2_1_left, ln1_1_n2_1_right, 
            ln1_2_n2_1_left, ln1_2_n2_1_right,
            ln1_1_n2_2_left, ln1_1_n2_2_right,
            ln1_2_n2_2_left, ln1_2_n2_2_right,
            
            # Split lines between hidden and output layer
            ln2_1_n3_1_left, ln2_1_n3_1_right,
            ln2_2_n3_1_left, ln2_2_n3_1_right,
            ln2_1_n3_2_left, ln2_1_n3_2_right,
            ln2_2_n3_2_left, ln2_2_n3_2_right,
            
            # Thetas (weights) on the connections
            theta_1, theta_2, theta_3, theta_4,
            theta_5, theta_6, theta_7, theta_8
        )
        
        s_network = t_network.copy()
        s_network.shift(RIGHT * 10)
        s_auxiliary_output_text.shift(RIGHT * 10)
        s_primary_output_text.shift(RIGHT * 10)

        s_f_s = VGroup(p29_to_32_to_manim_1.submobjects[153], p29_to_32_to_manim_1.submobjects[185]).next_to(s_primary_output_text, UP, buff=0.1)
        s_g_s = VGroup(p29_to_32_to_manim_1.submobjects[152], p29_to_32_to_manim_1.submobjects[186]).next_to(s_auxiliary_output_text, UP, buff=0.1)

        s_primary_output = VGroup(s_primary_output_text, s_f_s)
        s_auxiliary_output = VGroup(s_auxiliary_output_text, s_g_s)

        s_text = Text('STUDENT', font='Myriad Pro', font_size=48).next_to(VGroup(s_network[0], s_primary_output, s_auxiliary_output), UP, buff=0.5).set_color(CHILL_BROWN)
        s_model_parameters = Tex(
            r"\theta_S = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(VGroup(s_network[0], s_primary_output, s_auxiliary_output), DOWN)
        
        self.play(
            self.camera.frame.animate.move_to([ 5.1693025, -1.83217  ,  0.       ]).set_width(20)
        )
        
        self.wait(1)
            
        self.play(
            ReplacementTransform(t_network.copy(), s_network),
        )
        
        self.wait(1)
        
        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(p27_to_manim_6[0:13].copy(), s_primary_output_text),
                    ReplacementTransform(t_f_t[0].copy(), s_f_s[0]),
                    ReplacementTransform(t_f_t[1].copy(), s_f_s[1])
                ),
                AnimationGroup(
                    ReplacementTransform(p27_to_manim_6[13:28].copy(), s_auxiliary_output_text),
                    ReplacementTransform(t_g_t[0].copy(), s_g_s[0]),
                    ReplacementTransform(t_g_t[1].copy(), s_g_s[1])
                ),
                AnimationGroup(
                    ReplacementTransform(t_model_parameters[0].copy(), s_model_parameters[0]),
                    ReplacementTransform(t_model_parameters[1].copy(), s_model_parameters[1]),
                    ReplacementTransform(t_model_parameters[2:28].copy(), s_model_parameters[2:28]),
                ),
                lag_ratio=0.5,
                run_time=5
            )
        )
        
        self.wait(1)
        
        self.play(Write(s_text))
        
        self.wait(1)


        self.embed()