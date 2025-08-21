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
        
        model_parameters = Tex(
            r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
        ).set_color(FRESH_TAN).scale(0.75).next_to(n2_2, DOWN)

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
            FadeIn(VGroup(n1_1, n1_2))
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
            FadeIn(VGroup(n3_1, n3_2))
        )

        self.play(
            LaggedStart(
                group1, group2, group3, group4, group5, group6,
                lag_ratio=0.2,
                run_time=2
            )
        )
        
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
                run_time=2.5
            )
        )
        
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
            ReplacementTransform(theta_5.copy(), model_parameters[24:26]),
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
                run_time=2.5
            )
        )

        self.embed()