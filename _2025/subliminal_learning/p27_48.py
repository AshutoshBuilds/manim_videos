from manimlib import *
from MF_Tools import *

CHILL_BROWN = "#958a7a"
YELLOW = "#ffd35a"
YELLOW_FADE = "#7f6a2d"
BLUE = "#65c8d0"
GREEN = "#6e9671"
CHILL_GREEN = "#6c946f"
CHILL_BLUE = "#3d5c6f"
FRESH_TAN = "#dfd0b9"
COOL_GREEN = "#00a14b"
CYAN = "#00FFFF"

asset_dir_1='/Users/stephen/Stephencwelch Dropbox/welch_labs/subliminal_learning/graphics/to_manim'
full_path_to_working_dir='/Users/stephen/manim/videos/_2025/subliminal_learning'

def copy_frame_positioning_precise(frame):
    center = frame.get_center()
    height = frame.get_height()
    angles = frame.get_euler_angles()

    call = f"reorient("
    theta, phi, gamma = (angles / DEG)
    call += f"{theta}, {phi}, {gamma}"
    if any(center != 0):
        call += f", {tuple(center)}"
    if height != FRAME_HEIGHT:
        call += ", {:.2f}".format(height)
    call += ")"
    print(call)
    pyperclip.copy(call)


class P27_40(Scene):
    def construct(self):
        # p27_to_manim_1 = SVGMobject("p27_to_manim-01.svg")[1:].scale(4)
        p27_to_manim_2 = SVGMobject(asset_dir_1+"/p27_to_manim-02.svg")[1:].scale(4)
        p27_to_manim_3 = SVGMobject(asset_dir_1+"/p27_to_manim-03.svg")[1:].scale(4)
        # p27_to_manim_4 = SVGMobject("p27_to_manim-04.svg")[1:].scale(4)
        # p27_to_manim_5 = SVGMobject("p27_to_manim-05.svg")[1:].scale(4)
        p27_to_manim_6 = SVGMobject(asset_dir_1+"/p27_to_manim-06.svg")[1:].scale(4)
        # p27_to_manim_7 = SVGMobject("p27_to_manim-07.svg")[1:].scale(4)
        p28_to_manim_4 = SVGMobject(asset_dir_1+"/p28_to_manim-04.svg")[1:].scale(4)
        p29_to_32_to_manim_1 = SVGMobject(asset_dir_1+"/p29_to_32_to_manim-01.svg")[1:].scale(4)

        mnist_network = ImageMobject(full_path_to_working_dir+"/mnist_network.png")

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

        n2_1_text = VGroup(
            p27_to_manim_2.submobjects[17],
            p27_to_manim_2.submobjects[18],
            p27_to_manim_2.submobjects[19],
            p27_to_manim_2.submobjects[20],
        )
        n2_2_text = VGroup(
            p27_to_manim_2.submobjects[22],
            p27_to_manim_2.submobjects[23],
            p27_to_manim_2.submobjects[24],
            p27_to_manim_2.submobjects[25],
        )

        left_dimension = VGroup(
            p27_to_manim_3[0], p27_to_manim_3[1], p27_to_manim_3[2]
        ).next_to(VGroup(n1_1, n1_2), LEFT)
        top_dimension = VGroup(
            p27_to_manim_3[3], p27_to_manim_3[4], p27_to_manim_3[5]
        ).next_to(VGroup(n2_1, n3_1), UP)

        x1 = Tex(r"x_1").set_color(FRESH_TAN).move_to(n1_1.get_center())
        x2 = Tex(r"x_2").set_color(FRESH_TAN).move_to(n1_2.get_center())

        model_parameters = (
            Tex(
                r"\theta = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(n2_2, DOWN, buff=0.4)
        )

        model_parameters_text = (
            Text("Model Parameters*", font="Myriad Pro", font_size=24, weight=BOLD)
            .next_to(model_parameters, DOWN)
            .set_color(CHILL_BROWN)
        )

        def edge_point(c1, c2):
            c1_center = c1.get_center()
            c2_center = c2.get_center()
            direction = normalize(c2_center - c1_center)

            # ManimGL compatible approach - use get_corner instead of get_critical_point
            ul = c1.get_corner(UP + LEFT)
            ur = c1.get_corner(UP + RIGHT)
            dl = c1.get_corner(DOWN + LEFT)
            dr = c1.get_corner(DOWN + RIGHT)

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
                ReplacementTransform(line_part_2, split_line_2),
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

        ln1_1_n2_1_left, ln1_1_n2_1_right, theta_1_position = split_line(
            ln1_1_n2_1, theta_1.get_width() + 0.2, 0.5
        )
        theta_1.move_to(theta_1_position)
        ln1_1_n2_1_left.set_color(FRESH_TAN)
        ln1_1_n2_1_right.set_color(FRESH_TAN)

        ln1_2_n2_1_left, ln1_2_n2_1_right, theta_2_position = split_line(
            ln1_2_n2_1, theta_2.get_width() + 0.2, 0.75
        )
        theta_2.move_to(theta_2_position)
        ln1_2_n2_1_left.set_color(FRESH_TAN)
        ln1_2_n2_1_right.set_color(FRESH_TAN)

        ln1_1_n2_2_left, ln1_1_n2_2_right, theta_3_position = split_line(
            ln1_1_n2_2, theta_3.get_width() + 0.2, 0.75
        )
        theta_3.move_to(theta_3_position)
        ln1_1_n2_2_left.set_color(FRESH_TAN)
        ln1_1_n2_2_right.set_color(FRESH_TAN)

        ln1_2_n2_2_left, ln1_2_n2_2_right, theta_4_position = split_line(
            ln1_2_n2_2, theta_4.get_width() + 0.2, 0.5
        )
        theta_4.move_to(theta_4_position)
        ln1_2_n2_2_left.set_color(FRESH_TAN)
        ln1_2_n2_2_right.set_color(FRESH_TAN)

        ln2_1_n3_1_left, ln2_1_n3_1_right, theta_5_position = split_line(
            ln2_1_n3_1, theta_5.get_width() + 0.2, 0.5
        )
        theta_5.move_to(theta_5_position)
        ln2_1_n3_1_left.set_color(FRESH_TAN)
        ln2_1_n3_1_right.set_color(FRESH_TAN)

        ln2_2_n3_1_left, ln2_2_n3_1_right, theta_6_position = split_line(
            ln2_2_n3_1, theta_6.get_width() + 0.2, 0.75
        )
        theta_6.move_to(theta_6_position)
        ln2_2_n3_1_left.set_color(FRESH_TAN)
        ln2_2_n3_1_right.set_color(FRESH_TAN)

        ln2_1_n3_2_left, ln2_1_n3_2_right, theta_7_position = split_line(
            ln2_1_n3_2, theta_7.get_width() + 0.2, 0.75
        )
        theta_7.move_to(theta_7_position)
        ln2_1_n3_2_left.set_color(CHILL_BROWN)
        ln2_1_n3_2_right.set_color(CHILL_BROWN)

        ln2_2_n3_2_left, ln2_2_n3_2_right, theta_8_position = split_line(
            ln2_2_n3_2, theta_8.get_width() + 0.2, 0.5
        )
        theta_8.move_to(theta_8_position)
        ln2_2_n3_2_left.set_color(CHILL_BROWN)
        ln2_2_n3_2_right.set_color(CHILL_BROWN)

        group1 = AnimationGroup(FadeIn(VGroup(n1_1, n1_2, x1, x2)))

        group2 = AnimationGroup(
            GrowArrow(ln1_1_n2_1),
            GrowArrow(ln1_1_n2_2),
            GrowArrow(ln1_2_n2_1),
            GrowArrow(ln1_2_n2_2),
        )

        group3 = AnimationGroup(FadeIn(VGroup(n2_1, n2_2)))

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
            GrowFromEdge(n3_2_arrow, LEFT),
        )

        self.wait()
        self.play(
            LaggedStart(
                group1,
                group2,
                group3,
                group4,
                group5,
                group6,
                lag_ratio=0.2,
                run_time=2.5,
            )
        )

        self.wait(1)

        self.add(left_dimension, top_dimension)

        self.wait(1)

        theta1_group = AnimationGroup(
            animate_split_line(ln1_1_n2_1, ln1_1_n2_1_left, ln1_1_n2_1_right, 0.5),
            Write(theta_1),
            lag_ratio=0.5,
        )

        theta2_group = AnimationGroup(
            animate_split_line(ln1_2_n2_1, ln1_2_n2_1_left, ln1_2_n2_1_right, 0.75),
            Write(theta_2),
            lag_ratio=0.5,
        )

        theta3_group = AnimationGroup(
            animate_split_line(ln1_1_n2_2, ln1_1_n2_2_left, ln1_1_n2_2_right, 0.75),
            Write(theta_3),
            lag_ratio=0.5,
        )

        theta4_group = AnimationGroup(
            animate_split_line(ln1_2_n2_2, ln1_2_n2_2_left, ln1_2_n2_2_right, 0.5),
            Write(theta_4),
            lag_ratio=0.5,
        )

        theta5_group = AnimationGroup(
            animate_split_line(ln2_1_n3_1, ln2_1_n3_1_left, ln2_1_n3_1_right, 0.5),
            Write(theta_5),
            lag_ratio=0.5,
        )

        theta6_group = AnimationGroup(
            animate_split_line(ln2_2_n3_1, ln2_2_n3_1_left, ln2_2_n3_1_right, 0.75),
            Write(theta_6),
            lag_ratio=0.5,
        )

        theta7_group = AnimationGroup(
            animate_split_line(ln2_1_n3_2, ln2_1_n3_2_left, ln2_1_n3_2_right, 0.75),
            Write(theta_7),
            lag_ratio=0.5,
        )

        theta8_group = AnimationGroup(
            animate_split_line(ln2_2_n3_2, ln2_2_n3_2_left, ln2_2_n3_2_right, 0.5),
            Write(theta_8),
            lag_ratio=0.5,
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
                run_time=5,
            )
        )

        self.wait(1)

        model_parameters_1 = AnimationGroup(FadeIn(model_parameters[0:3]))

        model_parameters_2 = AnimationGroup(
            ReplacementTransform(theta_1.copy(), model_parameters[3:5]),
            FadeIn(model_parameters[5]),
        )

        model_parameters_3 = AnimationGroup(
            ReplacementTransform(theta_2.copy(), model_parameters[6:8]),
            FadeIn(model_parameters[8]),
        )

        model_parameters_4 = AnimationGroup(
            ReplacementTransform(theta_3.copy(), model_parameters[9:11]),
            FadeIn(model_parameters[11]),
        )

        model_parameters_5 = AnimationGroup(
            ReplacementTransform(theta_4.copy(), model_parameters[12:14]),
            FadeIn(model_parameters[14]),
        )

        model_parameters_6 = AnimationGroup(
            ReplacementTransform(theta_5.copy(), model_parameters[15:17]),
            FadeIn(model_parameters[17]),
        )

        model_parameters_7 = AnimationGroup(
            ReplacementTransform(theta_6.copy(), model_parameters[18:20]),
            FadeIn(model_parameters[20]),
        )

        model_parameters_8 = AnimationGroup(
            ReplacementTransform(theta_7.copy(), model_parameters[21:23]),
            FadeIn(model_parameters[23]),
        )

        model_parameters_9 = AnimationGroup(
            ReplacementTransform(theta_8.copy(), model_parameters[24:26]),
            FadeIn(model_parameters[26]),
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
                run_time=5,
            )
        )

        self.wait(1)

        self.camera.frame.save_state()

        self.play(FadeIn(model_parameters_text))

        self.wait(1)

        self.play(
            self.camera.frame.animate.move_to(
                [3, 0, 0]
            ),  # TODO: Update camera position
            FadeOut(top_dimension),
            FadeOut(left_dimension),
            FadeOut(model_parameters_text),
            run_time=2,
        )

        self.wait(1)

        p27_to_manim_6[29].next_to(n3_1_arrow, RIGHT, buff=0.1)
        p27_to_manim_6[28].next_to(n3_2_arrow, RIGHT, buff=0.1)

        p27_to_manim_6[0:13].next_to(p27_to_manim_6[29], DOWN, buff=0.1)
        p27_to_manim_6[13:28].next_to(p27_to_manim_6[28], DOWN, buff=0.1)

        primary_output = VGroup(p27_to_manim_6[29], p27_to_manim_6[0:13])
        auxiliary_output = VGroup(p27_to_manim_6[28], p27_to_manim_6[13:28])

        outputs = VGroup(primary_output, auxiliary_output)

        """self.play(FadeIn(primary_output), FadeIn(auxiliary_output), model_parameters.animate.move_to([ 2.81398236e-01, -1.57497560e+00, -2.99721055e-17]))

        self.wait(1)"""

        # p28_to_manim_4.move_to([(primary_output.get_x() + p28_to_manim_4.get_width()/2) - (p28_to_manim_4.submobjects[1].get_width()/2), 0, 0])

        p28_to_manim_4.move_to(
            [
                (primary_output.get_x() + p28_to_manim_4.get_width() / 2)
                - (p28_to_manim_4.submobjects[1].get_width() / 2),
                0,
                0,
            ]
        )

        top_arrow = VGroup(
            p28_to_manim_4.submobjects[0],
            p28_to_manim_4.submobjects[1],
            p28_to_manim_4.submobjects[2],
        )
        bottom_arrow = VGroup(
            p28_to_manim_4.submobjects[3],
            p28_to_manim_4.submobjects[4],
            p28_to_manim_4.submobjects[5],
        )

        top_outputs = outputs.get_top()[1]
        bottom_outputs = outputs.get_bottom()[1]
        bottom_submob1 = p28_to_manim_4.submobjects[1].get_bottom()[1]
        top_submob5 = p28_to_manim_4.submobjects[5].get_top()[1]
        vertical_shift = (
            top_outputs - bottom_submob1 - top_submob5 + bottom_outputs
        ) / 2

        # Apply the shift
        p28_to_manim_4.shift([0, vertical_shift, 0])

        # MATH TIME ;(
        """
        p28_to_manim_4.get_width() = 5.860444245515045
        was about to pull out the ti84 for this lol
        """

        mnist_network.move_to([7.325, -0.175, 0])  # TODO: Adjust position

        self.play(FadeIn(mnist_network))

        self.wait(1)

        self.play(
            LaggedStart(FadeIn(primary_output), FadeIn(top_arrow), lag_ratio=0.75)
        )

        self.wait(1)

        self.play(
            LaggedStart(FadeIn(auxiliary_output), FadeIn(bottom_arrow), lag_ratio=0.75)
        )

        # when mnist gets updated
        # self.play(FadeIn(mnist_network))

        self.wait(1)

        # P29

        self.play(
            self.camera.frame.animate.move_to([0, 0, 0]),
            FadeOut(mnist_network),
            FadeOut(p28_to_manim_4),
            run_time=2,
        )

        self.wait(3)

        t_network_copy = VGroup(
            # Input neurons with labels
            n1_1.copy(),
            n1_2.copy(),
            x1.copy(),
            x2.copy(),
            # Hidden layer neurons and their activation functions
            n2_1.copy(),
            n2_2.copy(),
            n2_1_graph.copy(),
            n2_2_graph.copy(),
            n2_1_text.copy(),
            n2_2_text.copy(),
            # Output layer neurons and arrows
            n3_1.copy(),
            n3_2.copy(),
            n3_1_arrow.copy(),
            n3_2_arrow.copy(),
            # Split lines between input and hidden layer
            ln1_1_n2_1_left.copy(),
            ln1_1_n2_1_right.copy(),
            ln1_2_n2_1_left.copy(),
            ln1_2_n2_1_right.copy(),
            ln1_1_n2_2_left.copy(),
            ln1_1_n2_2_right.copy(),
            ln1_2_n2_2_left.copy(),
            ln1_2_n2_2_right.copy(),
            # Split lines between hidden and output layer
            ln2_1_n3_1_left.copy(),
            ln2_1_n3_1_right.copy(),
            ln2_2_n3_1_left.copy(),
            ln2_2_n3_1_right.copy(),
            ln2_1_n3_2_left.copy(),
            ln2_1_n3_2_right.copy(),
            ln2_2_n3_2_left.copy(),
            ln2_2_n3_2_right.copy(),
            # Thetas (weights) on the connections
            theta_1.copy(),
            theta_2.copy(),
            theta_3.copy(),
            theta_4.copy(),
            theta_5.copy(),
            theta_6.copy(),
            theta_7.copy(),
            theta_8.copy(),
            model_parameters.copy(),
            auxiliary_output.copy(),
            primary_output.copy(),
        )


        # Assign individual variables for each element in the VGroup
        s_n1_1 = t_network_copy[0]
        s_n1_2 = t_network_copy[1]
        s_x1 = t_network_copy[2]
        s_x2 = t_network_copy[3]
        s_n2_1 = t_network_copy[4]
        s_n2_2 = t_network_copy[5]
        s_n2_1_graph = t_network_copy[6]
        s_n2_2_graph = t_network_copy[7]
        s_n2_1_text = t_network_copy[8]
        s_n2_2_text = t_network_copy[9]
        s_n3_1 = t_network_copy[10]
        s_n3_2 = t_network_copy[11]
        s_n3_1_arrow = t_network_copy[12]
        s_n3_2_arrow = t_network_copy[13]
        s_ln1_1_n2_1_left = t_network_copy[14]
        s_ln1_1_n2_1_right = t_network_copy[15]
        s_ln1_2_n2_1_left = t_network_copy[16]
        s_ln1_2_n2_1_right = t_network_copy[17]
        s_ln1_1_n2_2_left = t_network_copy[18]
        s_ln1_1_n2_2_right = t_network_copy[19]
        s_ln1_2_n2_2_left = t_network_copy[20]
        s_ln1_2_n2_2_right = t_network_copy[21]
        s_ln2_1_n3_1_left = t_network_copy[22]
        s_ln2_1_n3_1_right = t_network_copy[23]
        s_ln2_2_n3_1_left = t_network_copy[24]
        s_ln2_2_n3_1_right = t_network_copy[25]
        s_ln2_1_n3_2_left = t_network_copy[26]
        s_ln2_1_n3_2_right = t_network_copy[27]
        s_ln2_2_n3_2_left = t_network_copy[28]
        s_ln2_2_n3_2_right = t_network_copy[29]
        s_theta_1 = t_network_copy[30]
        s_theta_2 = t_network_copy[31]
        s_theta_3 = t_network_copy[32]
        s_theta_4 = t_network_copy[33]
        s_theta_5 = t_network_copy[34]
        s_theta_6 = t_network_copy[35]
        s_theta_7 = t_network_copy[36]
        s_theta_8 = t_network_copy[37]
        s_model_parameters = t_network_copy[38]
        s_auxiliary_output = t_network_copy[39]
        s_primary_output = t_network_copy[40]

        self.play(
            # self.camera.frame.animate.set_width(16).move_to([3.9, -1.4, 0]),
            self.frame.animate.reorient(0.0, 0.0, 0.0, (3.73, -1.84, 0.0), 8.74), #SW little different framing
            run_time=3,  # TODO: Find best camera position
        )

        self.wait(1)

        self.play(t_network_copy.animate.shift(RIGHT * 7), run_time=2.5)

        self.wait(1)

        t_model_parameters = (
            Tex(
                r"\theta_T = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(VGroup(n1_1, auxiliary_output), DOWN, buff=0.4)
        )

        t_f_t = (
            VGroup(
                p29_to_32_to_manim_1.submobjects[76],
                p29_to_32_to_manim_1.submobjects[183],
            )
            .set_color(FRESH_TAN)
            .move_to(primary_output[0].get_center())
        )

        t_g_t = (
            VGroup(
                p29_to_32_to_manim_1.submobjects[75],
                p29_to_32_to_manim_1.submobjects[184],
            )
            .set_color(CHILL_BROWN)
            .move_to(auxiliary_output[0].get_center())
        )

        def create_spaced_text(text_string, font, font_size, spacing=0.3, **kwargs):
            letters = []
            for i, letter in enumerate(text_string):
                letter_obj = Text(letter, font=font, font_size=font_size, **kwargs)
                if i > 0:
                    letter_obj.next_to(letters[-1], RIGHT, buff=spacing)
                letters.append(letter_obj)
            return VGroup(*letters)


        teacher_text = (
            # Text("TEACHER", font="Myriad Pro", font_size=32, weight=BOLD)
            create_spaced_text("TEACHER", font="Myriad Pro", font_size=32, spacing=0.1, weight=BOLD)
            .next_to(
                VGroup(n1_1, primary_output, auxiliary_output, t_model_parameters),
                UP,
                buff=0.5,
            )
            .set_color(CHILL_BROWN)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(
                        model_parameters[1:27], t_model_parameters[2:28]
                    ),
                    ReplacementTransform(model_parameters[0], t_model_parameters[0]),
                    FadeIn(t_model_parameters[1]),
                    ReplacementTransform(primary_output[0], t_f_t[0]),
                    FadeIn(t_f_t[1]),
                    ReplacementTransform(auxiliary_output[0], t_g_t[0]),
                    FadeIn(t_g_t[1]),
                ),
                Write(teacher_text),
                lag_ratio=0.8,
                run_time=2,
            )
        )

        self.wait(1)

        s_auxiliary_output_text = auxiliary_output.copy()[1]
        s_primary_output_text = primary_output.copy()[1]

        s_f_s = (
            VGroup(
                p29_to_32_to_manim_1.submobjects[153],
                p29_to_32_to_manim_1.submobjects[185],
            )
            .next_to(s_primary_output_text, UP, buff=0.1)
            .shift(RIGHT * 7)
        )
        s_g_s = (
            VGroup(
                p29_to_32_to_manim_1.submobjects[152],
                p29_to_32_to_manim_1.submobjects[186],
            )
            .next_to(s_auxiliary_output_text, UP, buff=0.1)
            .shift(RIGHT * 7)
        )

        s_model_parameters = (
            Tex(
                r"\theta_S = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8]"
            )
            .set_color(FRESH_TAN)
            .scale(0.75)
            .next_to(VGroup(t_network_copy[1], t_network_copy[39]), DOWN, buff=0.4)
        )

        # s_text = (
        #     Text("STUDENT", font="Myriad Pro", font_size=32, weight=BOLD, t2s={"letter_spacing": 10})
        #     .next_to(VGroup(t_network_copy[0], t_network_copy[40]), UP, buff=0.5)
        #     .set_color(CHILL_BROWN)
        # )


        s_text = (
            create_spaced_text("STUDENT", font="Myriad Pro", font_size=32, spacing=0.1, weight=BOLD)
            .next_to(VGroup(t_network_copy[0], t_network_copy[40]), UP, buff=0.5)
            .set_color(CHILL_BROWN)
                )

        self.play(
            LaggedStart(
                AnimationGroup(
                    ReplacementTransform(
                        t_network_copy[38][1:27], s_model_parameters[2:28]
                    ),
                    ReplacementTransform(t_network_copy[38][0], s_model_parameters[0]),
                    FadeIn(s_model_parameters[1]),
                    ReplacementTransform(t_network_copy[40][0], s_f_s[0]),
                    FadeIn(s_f_s[1]),
                    ReplacementTransform(t_network_copy[39][0], s_g_s[0]),
                    FadeIn(s_g_s[1]),
                ),
                Write(s_text),
                lag_ratio=0.8,
                run_time=2,
            )
        )

        self.wait(1)

        t_weight_eq = (
            Tex(r"\theta_T = \theta_T^0 + \Delta \theta_T")
            .move_to(t_model_parameters.get_center())
            .shift(DOWN * 2.75)
        )
        t_weight_text = (  # TODO: Maybe implement
            Text("Teacher weight\nupdate step", font="Myriad Pro", font_size=24)
            .next_to(t_weight_eq, RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )

        s_weight_eq = (
            Tex(r"\theta_S = \theta_S^0 + \Delta \theta_S")
            .move_to(s_model_parameters.get_center())
            .shift(DOWN * 2.75)
        )
        s_weight_text = (  # TODO: Maybe implement
            Text("Student weight\nupdate step", font="Myriad Pro", font_size=24)
            .next_to(s_weight_eq, RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )

        equal_weights = Tex(r"\theta_T^0 = \theta_S^0").move_to([3.5, -3, 0])
        equal_weights_text = (  # TODO: Maybe implement
            Text(
                "Teacher and student\nstart with same weights",
                font="Myriad Pro",
                font_size=24,
            )
            .next_to(equal_weights, RIGHT, buff=0.1)
            .set_color(CHILL_BROWN)
        )


        self.play(
            ReplacementTransform(
                t_model_parameters[0:2].copy(),
                VGroup(equal_weights[0], equal_weights[2]),
            ),
            ReplacementTransform(
                s_model_parameters[0:2].copy(),
                VGroup(equal_weights[4], equal_weights[6]),
            ),
            run_time=2,
        )

        self.wait(1)

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(equal_weights[1]),
                    FadeIn(equal_weights[5]),
                ),
                FadeIn(equal_weights[3]),
                lag_ratio=0.6,
                run_time=2,
            )
        )

        self.wait(1)

        self.play(
            FadeIn(t_weight_eq[0:3]),
            FadeIn(t_weight_eq[6:10]),
            ReplacementTransform(equal_weights[0:3].copy(), t_weight_eq[3:6]),
            run_time=1.5,
        )

        self.wait(1)

        self.play(
            FadeIn(s_weight_eq[0:3]),
            FadeIn(s_weight_eq[6:10]),
            ReplacementTransform(equal_weights[4:7].copy(), s_weight_eq[3:6]),
            run_time=1.5,
        )

        self.wait(2)

        # P33

        green_arrow = ImageMobject(
            full_path_to_working_dir+"/green_arrow.png"
        ).scale(  # TODO: If possible replace this with a SVG and make the position mathematically centered
            0.16517317500000003  # this is just a guess and check number
        )  # the svg version of this was bugged so i got sam to export a png version

        green_arrow.move_to((t_g_t.get_center() + s_g_s.get_center()) / 2)
        green_arrow.shift(DOWN * 0.6)

        small_t_weight_eq = t_weight_eq.copy().set_height(0.4).set_color(CHILL_BROWN)
        small_s_weight_eq = s_weight_eq.copy().set_height(0.4).set_color(CHILL_BROWN)
        small_equal_weights = (
            equal_weights.copy().set_height(0.4).set_color(CHILL_BROWN)
        )

        

        small_equations = VGroup(
            small_equal_weights, small_t_weight_eq, small_s_weight_eq
        ).arrange(DOWN)
        small_equations.move_to(
            [n1_2.get_left()[0] + small_equations.get_width() / 2, -4.5, 0]
        )
        small_equal_weights.move_to([-2.25, -3.8, 0]) #SW Manual bump here to make these guys left aligned. 

        self.play(
            ReplacementTransform(equal_weights, small_equal_weights),
            ReplacementTransform(t_weight_eq, small_t_weight_eq),
            ReplacementTransform(s_weight_eq, small_s_weight_eq),
            run_time=2,
        )

        self.wait(1)

        self.play(
            FadeOut(t_model_parameters),
            FadeOut(s_model_parameters),
            FadeOut(auxiliary_output[1]),
            FadeOut(t_network_copy[-2][1]),
        )

        # self.remove(small_s_weight_eq[-3:])

        self.wait(1)

        self.play(
            LaggedStart(
                VGroup(
                    n3_2,
                    n3_2_arrow,
                    theta_7,
                    theta_8,
                    ln2_2_n3_2_left,
                    ln2_2_n3_2_right,
                    ln2_1_n3_2_left,
                    ln2_1_n3_2_right,
                    t_g_t,
                ).animate.set_color(COOL_GREEN),
                FadeIn(green_arrow),
                VGroup(
                    t_network_copy[11],  # n3_2.copy()
                    t_network_copy[13],  # n3_2_arrow.copy()
                    t_network_copy[36],  # theta_7.copy()
                    t_network_copy[37],  # theta_8.copy()
                    t_network_copy[28],  # ln2_2_n3_2_left.copy()
                    t_network_copy[29],  # ln2_2_n3_2_right.copy()
                    t_network_copy[26],  # ln2_1_n3_2_left.copy()
                    t_network_copy[27],  # ln2_1_n3_2_right.copy()
                    s_g_s,
                ).animate.set_color(COOL_GREEN),
                lag_ratio=0.8,
                run_time=3.5,
            )
        )

        self.wait(1)

        squared_error = (
            Tex(r"L_S = \frac{1}{2}(g_T - g_S)^2").scale(0.9).next_to(green_arrow, DOWN)
        )

        squared_error[7:9].set_color(COOL_GREEN)
        squared_error[10:12].set_color(COOL_GREEN)

        self.play(
            FadeIn(squared_error[0:7]),
            FadeIn(squared_error[12:14]),
            FadeIn(squared_error[9]),
            ReplacementTransform(t_g_t.copy(), squared_error[7:9]),
            ReplacementTransform(s_g_s.copy(), squared_error[10:12]),
            run_time=2,
        )

        self.wait(1)

        # P34
        # SW jumping in here, ah man almost missed that this was a delta but is supposed to be a nabla
        # easy mistake!

        gradient_descent_eq = (
            Tex(r"\Delta \theta_S = - \alpha \nabla_\theta L_S")
            # .scale(0.9)
            .next_to(squared_error, DOWN, buff=0.6)
        )

        gradient_descent_eq.shift([-0.2, 0, 0]) #Nudge for better horizontal aligmeent. 

        gradient_descent_eq[6:8].set_color(YELLOW)

        #SW -> Ok can i move over the Delta Theta S from the subscript equation - that would really help!
        delta_theta_s_copy=small_s_weight_eq[-3:].copy()


        self.wait()
        self.play(ReplacementTransform(delta_theta_s_copy, gradient_descent_eq[:3]))
        self.play(Write(gradient_descent_eq[3:]))

        self.wait(1)

        # SW Lets take a crack at adding in text & arrows and see how it feels
        p33_to_manim_2 = SVGMobject(asset_dir_1+"/p33_40_to_manim-02.svg")[1:].scale(5)

        p33_to_manim_2.move_to([6.8, -3.55, 0])

        self.play(Write(p33_to_manim_2[:21])) #Label 1
        self.add(p33_to_manim_2[71]) #Arrer
        self.wait()

        self.play(Write(p33_to_manim_2[58:70])) #Label 2
        self.add(p33_to_manim_2[70]) #Arrer
        self.wait()

        self.play(Write(p33_to_manim_2[42:58])) #Label 3
        self.add(p33_to_manim_2[72]) #Arrer
        self.wait()

        self.play(Write(p33_to_manim_2[21:42])) #Label 4
        self.add(p33_to_manim_2[73]) #Arrer
        self.wait()
        # self.play(FadeIn(p33_to_manim_2[:21]))

        # self.add(p33_to_manim_2[72])

        # self.remove(p33_to_manim_2)
        self.wait()


        # P35
        funky_arrow_1 = SVGMobject(asset_dir_1+"/p33_40_to_manim-03.svg")[1:].scale(5)
        funky_arrow_1.move_to([7.4, -4, 0])
        gradient_vector = (
            Tex(
                (
                    r"\left[ \frac{\partial L_S}{\partial \theta_1},"
                    r"\frac{\partial L_S}{\partial \theta_2},"
                    r"\cdots,"
                    r"\frac{\partial L_S}{\partial \theta_8} \right]"
                )
            )
            .set_color(YELLOW)
            .scale(0.8)
            .next_to(gradient_descent_eq, DOWN, buff=0.7)
        )
        gradient_vector.shift(([0.64, 0.0, 0])) #lil nudge

        self.remove(p33_to_manim_2[:21], p33_to_manim_2[42:58], p33_to_manim_2[71], p33_to_manim_2[72])
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(gradient_vector[0]),
                    FadeIn(gradient_vector[1]),
                    FadeIn(gradient_vector[4]),
                    FadeIn(gradient_vector[5]),
                    FadeIn(gradient_vector[8]),
                    ReplacementTransform(
                        squared_error[0:2].copy().copy().copy().copy(),
                        gradient_vector[2:4],
                    ),
                    ReplacementTransform(s_theta_1.copy(), gradient_vector[6:8]),
                ),
                AnimationGroup(
                    FadeIn(gradient_vector[9]),
                    FadeIn(gradient_vector[12]),
                    FadeIn(gradient_vector[13]),
                    FadeIn(gradient_vector[16]),
                    ReplacementTransform(
                        squared_error[0:2].copy().copy().copy(), gradient_vector[10:12]
                    ),
                    ReplacementTransform(s_theta_2.copy(), gradient_vector[14:16]),
                    FadeIn(gradient_vector[17:21]),
                ),
                AnimationGroup(
                    FadeIn(gradient_vector[28]),
                    FadeIn(gradient_vector[21]),
                    FadeIn(gradient_vector[24]),
                    FadeIn(gradient_vector[25]),
                    ReplacementTransform(
                        squared_error[0:2].copy().copy(), gradient_vector[22:24]
                    ),
                    ReplacementTransform(s_theta_8.copy(), gradient_vector[26:28]),
                ),
                lag_ratio=0.5,
                run_time=5,
            )
        )
        
        self.add(funky_arrow_1)
        self.wait()

        #P36 Let's Go
        self.play(FadeOut(gradient_vector), FadeOut(funky_arrow_1), FadeOut(p33_to_manim_2[58:70]),
                  FadeOut(p33_to_manim_2[21:42]), FadeOut(p33_to_manim_2[70]), FadeOut(p33_to_manim_2[73]))
        self.wait()

        #Ok a few things at once, move grad equation to the left, add a second part, brining over needed compoenents
        #Leg me kinda work backwards here...

        # gradient_descent_eq.move_to([6.303-2, -3.425,  0]) #Animate this

        gradient_descent_eq_2 = Tex(r" = - \alpha \nabla_\theta \bigg[ \frac{1}{2}(g_T - g_S)^2  \bigg]")
        gradient_descent_eq_2.move_to([6.303+2.3, -3.425,  0])
        # self.add(gradient_descent_eq_2[:5])
        gradient_descent_eq_2[5:].scale(0.9) #Gets kinda big otherwise?
        gradient_descent_eq_2[5:].shift([-0.2, 0, 0])
        # self.add(gradient_descent_eq_2[5:])

        gradient_descent_eq_2[3:5].set_color(YELLOW)
        gradient_descent_eq_2[10:12].set_color(COOL_GREEN)
        gradient_descent_eq_2[13:15].set_color(COOL_GREEN)

        # gradient_descent_eq_2.scale(0.9) #Too inconsistent or ok?
        
        self.wait()
        self.play(gradient_descent_eq.animate.move_to([6.303-2, -3.425,  0]))
        self.play(FadeIn(gradient_descent_eq_2[0]), 
                  ReplacementTransform(gradient_descent_eq[4:8].copy(),gradient_descent_eq_2[1:5])) #Might want to make an explicity copy here for cleanup!
        self.play(
                # FadeIn(gradient_descent_eq_2[5]), FadeIn(gradient_descent_eq_2[-1]), 
                  ReplacementTransform(squared_error[3:].copy(), gradient_descent_eq_2[6:-1]))
        self.add(gradient_descent_eq_2[5], gradient_descent_eq_2[-1]) #Add dem berkets
        self.wait()


        # Script goes kinda fast here, so I'm wondering if want to move down a copy while moving down the 2?
        # Could also obviously adjust the script. Hmm. 
        # Hmm how do I want to do this...
        # Ok yeah I think I I want to move a copy down first, let me see here...

        gradient_descent_eq_2_copy_a=gradient_descent_eq_2[:3].copy()
        gradient_descent_eq_2_copy_b=gradient_descent_eq_2[6:-1].copy()

        self.wait()
        self.add(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)
        self.play(gradient_descent_eq_2_copy_a.animate.shift([0, -1, 0]),
                    gradient_descent_eq_2_copy_b.animate.shift([0, -1, 0]))

        # gradient_descent_eq_2_copy_b[-1].next_to(gradient_descent_eq_2_copy_b[1], LEFT, buff=0.2)
        # gradient_descent_eq_2_copy_b[-1].scale(1.5)
        self.wait()
        self.play(gradient_descent_eq_2_copy_b[-1].animate.scale(1.36).next_to(gradient_descent_eq_2_copy_b[1], LEFT, buff=0.2))
        self.wait()

        #Ah I don't want to draw a diagonal line but it will look nice...here we go...
        cross_out_line_1=Line([ 7.8, -4.25,  0], [ 8.5, -4.85,  0])
        cross_out_line_1.set_stroke(color=YELLOW, width=4)

        self.play(ShowCreation(cross_out_line_1))
        self.play(FadeOut(cross_out_line_1), FadeOut(gradient_descent_eq_2_copy_b[-1]), 
            FadeOut(gradient_descent_eq_2_copy_b[:3]))
            # gradient_descent_eq_2_copy_b[3:].animate.shift([-1.1, 0, 0]))
        self.play(gradient_descent_eq_2_copy_b[3:-1].animate.shift([-1.05, 0, 0]))
        self.wait()

        gs_copy=gradient_descent_eq_2_copy_b[-4:-2].copy()
        nabla_copy=gradient_descent_eq_2[3:5].copy()
        self.add(nabla_copy, gs_copy)
        
        # nabla_copy.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.1).shift([0, -0.04, 0])
        # gs_copy.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.7).shift([0.03, -0.1, 0])
        self.wait()
        self.play(nabla_copy.animate.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.1).shift([0, -0.04, 0]),
                 gs_copy.animate.next_to(gradient_descent_eq_2_copy_b[3:-1], RIGHT, buff=0.7).shift([0.03, -0.1, 0]),
                 )
        self.wait()


        gradient_vector_2 = (
            Tex(
                (
                    r"\left[ \frac{\partial g_S}{\partial \theta_1},"
                    r"\frac{\partial g_S}{\partial \theta_2},"
                    r"\cdots,"
                    r"\frac{\partial g_S}{\partial \theta_8} \right]"
                )
            )
            .set_color(YELLOW)
            .scale(0.8)
            .next_to(gradient_descent_eq_2_copy_b, DOWN, buff=0.3)
            .shift([0.6, 0, 0])
        )

        gradient_vector_2[2:4].set_color(COOL_GREEN)
        gradient_vector_2[10:12].set_color(COOL_GREEN)
        gradient_vector_2[22:24].set_color(COOL_GREEN)


        # Ok, now how did pranav to the cool animation last time?
        # Also, let me go ahead and add the random arrow
        funky_arrow_2 = SVGMobject(asset_dir_1+"/p33_40_to_manim-04.svg")[1:].scale(5)
        funky_arrow_2.move_to([9.9, -4.95, 0])       

        self.wait()
        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(gradient_vector_2[0]),
                    FadeIn(gradient_vector_2[1]),
                    FadeIn(gradient_vector_2[4]),
                    FadeIn(gradient_vector_2[5]),
                    FadeIn(gradient_vector_2[8]),
                    ReplacementTransform(
                        gs_copy.copy().copy().copy().copy(),
                        gradient_vector_2[2:4],
                    ),
                    ReplacementTransform(s_theta_1.copy(), gradient_vector_2[6:8]),
                ),
                AnimationGroup(
                    FadeIn(gradient_vector_2[9]),
                    FadeIn(gradient_vector_2[12]),
                    FadeIn(gradient_vector_2[13]),
                    FadeIn(gradient_vector_2[16]),
                    ReplacementTransform(
                        gs_copy.copy().copy().copy(), gradient_vector_2[10:12]
                    ),
                    ReplacementTransform(s_theta_2.copy(), gradient_vector_2[14:16]),
                    FadeIn(gradient_vector_2[17:21]),
                ),
                AnimationGroup(
                    FadeIn(gradient_vector_2[28]),
                    FadeIn(gradient_vector_2[21]),
                    FadeIn(gradient_vector_2[24]),
                    FadeIn(gradient_vector_2[25]),
                    ReplacementTransform(
                        gs_copy.copy().copy(), gradient_vector_2[22:24]
                    ),
                    ReplacementTransform(s_theta_8.copy(), gradient_vector_2[26:28]),
                ),
                lag_ratio=0.5,
                run_time=5,
            )
        )
        self.add(funky_arrow_2)
        self.wait()

        # P37 - phew p36 was no joke lol
        # Alright first a little clean up
        # I don't want to end up with a frankenstien equation going foward, 
        # So let me see if I can do some decent clean up here...

        gradient_descent_eq_3 = Tex(r" = - \alpha  (g_T - g_S) \nabla_\theta g_S")
        gradient_descent_eq_3.move_to([6.303+2.05, -3.425,  0])
        gradient_descent_eq_3[-4:-2].set_color(YELLOW)
        gradient_descent_eq_3[-2:].set_color(COOL_GREEN)
        gradient_descent_eq_3[4:6].set_color(COOL_GREEN)
        gradient_descent_eq_3[7:9].set_color(COOL_GREEN)


        self.wait()
        self.play(FadeOut(funky_arrow_2), FadeOut(gradient_vector_2), FadeOut(gradient_descent_eq_2))
        self.play(ReplacementTransform(gradient_descent_eq_2_copy_a, gradient_descent_eq_3[:3]), 
                  ReplacementTransform(gradient_descent_eq_2_copy_b[3:-1], gradient_descent_eq_3[3:10]), 
                  ReplacementTransform(nabla_copy, gradient_descent_eq_3[10:12]),
                  ReplacementTransform(gs_copy, gradient_descent_eq_3[12:]))
        self.wait()


        # Hmm maybe I can kinda do a nice lag ratio for the S to zero transformation?
        # Hmm replaceing entire equation seems a bit problematic?

        squared_error_b = (
            Tex(r"L_S = \frac{1}{2}(g_T - g_0)^2").scale(0.9).move_to(squared_error, aligned_edge=LEFT)
        )

        squared_error_b[7:9].set_color(COOL_GREEN)
        squared_error_b[10:12].set_color(COOL_GREEN)

        gradient_descent_eq_3b = Tex(r" = - \alpha  (g_T - g_0) \nabla_\theta g_0")
        gradient_descent_eq_3b.move_to(gradient_descent_eq_3, aligned_edge=LEFT)
        gradient_descent_eq_3b[-4:-2].set_color(YELLOW)
        gradient_descent_eq_3b[-2:].set_color(COOL_GREEN)
        gradient_descent_eq_3b[4:6].set_color(COOL_GREEN)
        gradient_descent_eq_3b[7:9].set_color(COOL_GREEN)

        self.wait()
        self.play(ReplacementTransform(squared_error, squared_error_b), 
                  ReplacementTransform(gradient_descent_eq_3, gradient_descent_eq_3b), lag_ratio=0.5, run_time=3)
        self.wait()

        #Ok I think that will work for changing the g0s to gS. 


        ## --- P38 --- ##
        # Okie dokie artichhokie - now I need to create teh "bottom legend" in manim -> this will take a little time
        # and that's ok -> it will persist for the rest of the video. Tempted to try both equations and text - 
        # let's see how it goes...

        #Eitheer at the same time or before I adjust the legend, I need to reduce the size of the student equations and 
        # the green arrow -> let's try same time
        # Basically all making space for math on th teacher.


        # self.remove(green_arrow)
        # self.remove(small_equal_weights)
        # self.add(small_equal_weights)

        # small_equal_weights, 
        # small_t_weight_eq, 
        # small_s_weight_eq


        #Ok, make new equations or push them around? Maybe new ones?
        t_weight_eq_2 = Tex(r"\theta_T = \theta_T^0 + \Delta \theta_T", font_size=30).set_color(CHILL_BROWN)
        s_weight_eq_2 = Tex(r"\theta_S = \theta_S^0 + \Delta \theta_S", font_size=30).set_color(CHILL_BROWN)
        equal_weights_eq_2 = Tex(r"\theta_T^0 = \theta_S^0 = \theta^0", font_size=30).set_color(CHILL_BROWN)
        g_0_eq = Tex(r"g_0=g(X, \theta^0)", font_size=30).set_color(CHILL_BROWN)

        spacing=3.7; start=-2.5
        equal_weights_eq_2.move_to([start, -5.85, 0])
        t_weight_eq_2.move_to([start+spacing*1, -5.85, 0])
        s_weight_eq_2.move_to([start+spacing*2, -5.85, 0])
        g_0_eq.move_to([start+spacing*3, -5.85, 0])

        equal_weights_text_2 = (  # TODO: Maybe implement
            MarkupText(
                "Teacher and student\nstart with same weights",
                alignment="left",
                font="Myriad Pro",
                font_size=13,
            )
            .next_to(equal_weights_eq_2, RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        t_weight_text_2 = (  # TODO: Maybe implement
            MarkupText("Teacher weight\nupdate step", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(t_weight_eq_2, RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        s_weight_text_2 = (  # TODO: Maybe implement
            MarkupText("Student weight\nupdate step", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(s_weight_eq_2, RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )

        g_0_eq_text = (  # TODO: Maybe implement
            MarkupText("Initial auxiliary output \nbefore training", alignment="left", font="Myriad Pro", font_size=13)
            .next_to(g_0_eq, RIGHT, buff=0.15)
            .set_color(CHILL_BROWN)
        )


        # self.add(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2, g_0_eq)
        # self.add(equal_weights_text_2, t_weight_text_2, s_weight_text_2, g_0_eq_text)

        #Ok pretty good, now I need a horizontal line and the big animation!
        legend_line=Line([-3.7, -5.5, 0], [11.2, -5.5, 0], buff=0)
        legend_line.set_stroke(color=CHILL_BROWN, width=1)
        # self.add(legend_line)


        # self.remove(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2)


        # squared_error_b.scale(0.8) 
        # squared_error_b.move_to([6.9, -1.8, 0])

        student_line_2_eq_p38=VGroup(gradient_descent_eq, gradient_descent_eq_3b)
        # student_line_2_eq_p38.scale(0.8)
        # student_line_2_eq_p38.move_to([7.2, -2.6, 0])

        #Bring back in these labels
        auxiliary_output[1].set_color(COOL_GREEN)
        t_network_copy[-2][1].set_color(COOL_GREEN)

        #Now do all moves together
        self.wait()
        # self.remove(small_t_weight_eq, small_s_weight_eq, small_equal_weights)
        self.play(FadeOut(green_arrow), 
            squared_error_b.animate.scale(0.8).move_to([6.9, -1.8, 0]), 
            student_line_2_eq_p38.animate.scale(0.8).move_to([7.2, -2.6, 0]),
            ReplacementTransform(small_equal_weights, equal_weights_eq_2[:-3]), 
            ReplacementTransform(small_t_weight_eq, t_weight_eq_2),
            ReplacementTransform(small_s_weight_eq, s_weight_eq_2),
            )
        self.play(ShowCreation(legend_line), FadeIn(equal_weights_eq_2[-3:]), FadeIn(g_0_eq), 
                 FadeIn(equal_weights_text_2), FadeIn(t_weight_text_2), FadeIn(s_weight_text_2), FadeIn(g_0_eq_text), 
                 FadeIn(auxiliary_output[1]), FadeIn(t_network_copy[-2][1]))
        self.wait()

        # self.add(equal_weights_eq_2, t_weight_eq_2, s_weight_eq_2, g_0_eq)
        # self.add(equal_weights_text_2, t_weight_text_2, s_weight_text_2, g_0_eq_text)

        gt_question_eq=Tex("g_T= \: ?", font_size=56).set_color(COOL_GREEN)
        gt_question_eq.move_to([0, -2.3, 0])

        self.play(ReplacementTransform(t_g_t.copy(), gt_question_eq[:2]))
        self.play(Write(gt_question_eq[2:]))
        self.wait()


        p33_40_to_manim_05 = SVGMobject(asset_dir_1+"/p33_40_to_manim-05.svg")[1:].scale(4)
        
        p33_40_to_manim_05.move_to([0.2, -2.25, 0])
        g_0_eq_2=Tex("g_0", font_size=56).set_color(COOL_GREEN)
        g_0_eq_2.move_to([-0.8, -2.4, 0])


        self.remove(gt_question_eq[2:])
        self.play(gt_question_eq[:2].animate.shift([1.7, 0, 0]))
        self.play(ShowCreation(p33_40_to_manim_05[:-3]), Write(g_0_eq_2))
        self.wait()
        self.play(ReplacementTransform(t_weight_eq_2[-3:].copy(), p33_40_to_manim_05[-3:]))
        self.wait()


        #Ok let me kinda start with the end in midn here and get finalish equation setup first. 
        taylor_expansion_1=Tex(r"g_T \approx g_0 + \frac{\partial g_0}{\partial \theta_1} \Delta \theta_1 + ... ", font_size=50)
        taylor_expansion_1.move_to([0.1, -2.3, 0])
        taylor_expansion_1[:2].set_color(COOL_GREEN)
        taylor_expansion_1[3:5].set_color(COOL_GREEN)
        taylor_expansion_1[6:13].set_color(YELLOW)
        taylor_expansion_1[13:16].set_color(CYAN)

        # self.add(taylor_expansion_1)
        # self.remove(taylor_expansion_1)

        self.wait()
        self.play(FadeOut(p33_40_to_manim_05[:-3]))
        self.play(ReplacementTransform(g_0_eq_2, taylor_expansion_1[3:5]), 
                  ReplacementTransform(gt_question_eq[:2], taylor_expansion_1[:2]), 
                  ReplacementTransform(p33_40_to_manim_05[-3:], taylor_expansion_1[-7:-4]))
        self.play(FadeIn(taylor_expansion_1[2]), FadeIn(taylor_expansion_1[5]), 
                  FadeIn(taylor_expansion_1[6:13]), FadeIn(taylor_expansion_1[-4:]))

        self.wait()

        # I think especially when we get to the drawin here, it will be cool to 
        # possibly bring things down from the teacher model!

        p33_40_to_manim_06 = SVGMobject(asset_dir_1+"/p33_40_to_manim-06.svg")[1:].scale(4)
        p33_40_to_manim_07 = SVGMobject(asset_dir_1+"/p33_40_to_manim-07.svg")[1:].scale(4)
        p33_40_to_manim_08 = SVGMobject(asset_dir_1+"/p33_40_to_manim-08.svg")[1:].scale(4)
        p33_40_to_manim_09 = SVGMobject(asset_dir_1+"/p33_40_to_manim-09.svg")[1:].scale(4)
        p33_40_to_manim_06.move_to([0.9, -4.2, 0])
        p33_40_to_manim_07.move_to([1.15, -4.1, 0])
        p33_40_to_manim_08.move_to([0.88, -4.9, 0])
        p33_40_to_manim_09.move_to([0.65, -3.6, 0])
        taylor_expansion_plot_1 = Group(p33_40_to_manim_06, p33_40_to_manim_07, p33_40_to_manim_08, p33_40_to_manim_09)


        self.wait()
        self.play(FadeIn(p33_40_to_manim_06))
        self.play(Write(p33_40_to_manim_07[-1]), 
            ReplacementTransform(taylor_expansion_1[6:13].copy(), p33_40_to_manim_07[:-1]))
        self.wait()

        self.play(FadeIn(p33_40_to_manim_08[-2:]), ReplacementTransform(taylor_expansion_1[13:16].copy(), p33_40_to_manim_08[:-2]))
        self.wait()
        self.play(Write(p33_40_to_manim_09))
        self.wait()


        # Ok final bit of this big ole paragraph. 
        # Alright lets see here, let me start with the end again for this last move of p38. 
        p33_40_to_manim_10 = SVGMobject(asset_dir_1+"/p33_40_to_manim-10.svg")[1:].scale(4)
        taylor_expansion_2=Tex(r"g_T \approx g_0 + \frac{\partial g_0}{\partial \theta_1} \Delta \theta_1 + \frac{\partial g_0}{\partial \theta_2} \Delta \theta_2 + \frac{\partial g_0}{\partial \theta_3} \Delta \theta_3 +... ", font_size=36)
        taylor_expansion_2.move_to([0.1, -2.3, 0])
        taylor_expansion_2[:2].set_color(COOL_GREEN)
        taylor_expansion_2[3:5].set_color(COOL_GREEN)
        taylor_expansion_2[6:13].set_color(YELLOW)
        taylor_expansion_2[13:16].set_color(CYAN)
        taylor_expansion_2[17:24].set_color(YELLOW)
        taylor_expansion_2[24:27].set_color(CYAN)
        taylor_expansion_2[28:35].set_color(YELLOW)
        taylor_expansion_2[35:38].set_color(CYAN)

        # taylor_expansion_plot_1.scale(0.7)
        # taylor_expansion_plot_1.move_to([-1.2, -3.5, 0])
        p33_40_to_manim_10.move_to([1.55, -3.55, 0])


        #Animate then tweak positions and colorz. 
        self.wait()
        self.remove(taylor_expansion_1[-3:]) #Drop elipses
        self.play(ReplacementTransform(taylor_expansion_1[:-3], taylor_expansion_2[:len(taylor_expansion_1)-3]), 
                  taylor_expansion_plot_1.animate.scale(0.7).move_to([-1.2, -3.5, 0]))
        self.play(FadeIn(taylor_expansion_2[len(taylor_expansion_1)-3:]), FadeIn(p33_40_to_manim_10))

        self.wait()


        ## --- P39 let's go --- ## 
        taylor_expansion_3=Tex(r"g_T \approx g_0 + \nabla_{\theta} g_0 \cdot \Delta \theta_T", font_size=42)
        taylor_expansion_3.move_to([0.1, -2.3, 0])
        taylor_expansion_3[:2].set_color(COOL_GREEN)
        taylor_expansion_3[3:5].set_color(COOL_GREEN)
        taylor_expansion_3[6:8].set_color(YELLOW)
        taylor_expansion_3[8:10].set_color(COOL_GREEN)
        taylor_expansion_3[11:].set_color(CYAN)


        # self.add(taylor_expansion_3)


        self.wait()
        self.play(FadeOut(p33_40_to_manim_10), FadeOut(taylor_expansion_plot_1))
        self.remove(taylor_expansion_2[16], taylor_expansion_2[27], taylor_expansion_2[38:])
        self.play(ReplacementTransform(taylor_expansion_2[:6], taylor_expansion_3[:6]), 
                  ReplacementTransform(taylor_expansion_2[6:13], taylor_expansion_3[6:8]), 
                  ReplacementTransform(taylor_expansion_2[17:24], taylor_expansion_3[6:8]),
                  ReplacementTransform(taylor_expansion_2[28:35], taylor_expansion_3[6:8]),
                  ReplacementTransform(taylor_expansion_2[13:16], taylor_expansion_3[11:]), 
                  ReplacementTransform(taylor_expansion_2[24:27], taylor_expansion_3[11:]),
                  ReplacementTransform(taylor_expansion_2[35:38], taylor_expansion_3[11:]),
                  ReplacementTransform(taylor_expansion_2[3:5].copy(), taylor_expansion_3[8:10]))
        self.add(taylor_expansion_3[10])
        self.wait()

        # self.remove(taylor_expansion_2)
        # self.add(taylor_expansion_3)
        # self.add(taylor_expansion_3[8:10])
        p33_40_to_manim_11 = SVGMobject(asset_dir_1+"/p33_40_to_manim-11.svg")[1:].scale(4.2)
        p33_40_to_manim_11.move_to([1.1, -2.7, 0])

        self.play(FadeIn(p33_40_to_manim_11))
        self.wait()


        # self.remove(p33_40_to_manim_11)

        ### -- P40 -- ###
        




        self.wait(20)
        self.embed()




        # self.add(taylor_expansion_2)
        # self.add(p33_40_to_manim_10)


        # self.play(Write(p33_40_to_manim_07[-1]))
        # self.add(p33_40_to_manim_07[-1])

        # self.remove(taylor_expansion_plot_1)



        #In [4]: copy_frame_positioning_precise(self.frame)                                                                                                                          
        #reorient(0.0, 0.0, 0.0, (3.7340972, -1.8347028, 0.0), 8.74)  
        # reorient(0.0, 0.0, 0.0, (3.73, -1.84, 0.0), 8.74)  



        # self.add(gradient_descent_eq_3)
        # self.remove(gradient_descent_eq_2)
        # self.remove(gradient_descent_eq_2_copy_a)
        # self.remove(gradient_descent_eq_2_copy_b)
        # self.remove(nabla_copy)
        # self.remove(gs_copy)
        # self.remove(funky_arrow_2)
        # self.remove(gradient_vector_2)


        # self.add(funky_arrow_2)
        # self.add(gradient_vector_2)
        # self.remove(gradient_vector_2)




        # self.add(cross_out_line_1)
        # self.remove(cross_out_line_1)

        # self.play(gradient_descent_eq_2_copy_b[-1].animate


        # self.play(gradient_descent_eq_2_copy_b[-1].animate.move_to(gradient_descent_eq_2_copy_b[0].get_center()), 
        #           FadeOut(gradient_descent_eq_2_copy_b[0]))


        # self.add(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)

        # self.remove(gradient_descent_eq_2_copy_a, gradient_descent_eq_2_copy_b)

        # gradient_descent_eq_3 = Tex(r" = - \alpha (g_T - g_S)  \bigg]")
        # gradient_descent_eq_3.scale(0.9)


    #











