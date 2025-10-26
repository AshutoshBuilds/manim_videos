from manim_imports_ext import *
import numpy as np


class GradientDescentExplanation(Scene):
    def construct(self):
        # Set up colors and background
        self.camera.background_color = "#0f0f23"

        # Start immediately with an engaging hook - mathematical landscape
        # Create a beautiful mathematical background with multiple functions
        background_axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-2, 3, 1],
            axis_config={"color": BLUE_E, "stroke_width": 1, "include_tip": False}
        ).scale(0.8).to_edge(DOWN, buff=1).shift(DOWN * 0.5)

        # Multiple colorful function curves as background
        functions = [
            lambda x: 0.3 * x**2 + 0.1 * np.sin(3*x),
            lambda x: -0.2 * x**3 + 0.5 * x**2 - 0.1 * x,
            lambda x: 0.15 * np.sin(2*x) + 0.1 * np.cos(4*x),
        ]
        colors = [BLUE_D, GREEN_D, PURPLE_D]

        background_curves = VGroup()
        for func, color in zip(functions, colors):
            curve = background_axes.get_graph(func, color=color, stroke_width=2, x_range=[-3.5, 3.5])
            background_curves.add(curve)

        # Add some floating mathematical symbols
        symbols = VGroup(
            Text("∇", font_size=30, color=BLUE_C).move_to([-5, 2, 0]),
            Text("∫", font_size=25, color=GREEN_C).move_to([5, -1, 0]),
            Text("∑", font_size=28, color=PURPLE_C).move_to([-4, -2, 0]),
            Text("∂", font_size=26, color=YELLOW_C).move_to([4, 1.5, 0]),
        )

        # Fade in background elements immediately
        self.add(background_curves, symbols)

        # Dynamic title entrance with particle effect
        title_particles = VGroup(*[
            Dot(
                radius=0.02,
                color=YELLOW,
            ).move_to([
                np.random.uniform(-6, 6),
                np.random.uniform(-3, 3),
                0
            ]) for _ in range(20)
        ])

        title = Text("Gradient Descent", font_size=72, color=YELLOW, weight=BOLD)
        title_shadow = Text("Gradient Descent", font_size=72, color=BLACK).shift([0.02, -0.02, -0.1])

        # Animate title entrance with particles
        self.play(
            LaggedStart(*[FadeIn(particle, run_time=0.5) for particle in title_particles]),
            FadeIn(title_shadow, run_time=0.5),
            Write(title, run_time=1.5),
            run_time=2
        )

        subtitle = Text("Finding the Path to Optimization", font_size=32, color=BLUE_C)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(
            Write(subtitle, run_time=1),
            *[particle.animate.set_color(BLUE_B).scale(0.5) for particle in title_particles],
            run_time=1
        )
        self.wait(0.5)

        # Section 1: The Problem Setup
        # Transform title area into problem statement
        self.play(
            title.animate.scale(0.7).to_edge(UP, buff=0.3),
            subtitle.animate.scale(0.8).next_to(title, DOWN, buff=0.2),
            FadeOut(title_particles, run_time=0.8),
            run_time=1
        )

        # Create a dramatic problem setup
        problem_text = Text("The Challenge:", font_size=42, color=RED_C)
        problem_text.next_to(subtitle, DOWN, buff=0.8)

        problem_desc = Text("Find the lowest point in a complex landscape", font_size=32, color=WHITE)
        problem_desc.next_to(problem_text, DOWN, buff=0.4)

        # Visualize the problem with a complex function
        problem_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 4, 1],
            axis_config={"color": WHITE, "stroke_width": 2}
        ).scale(0.9).next_to(problem_desc, DOWN, buff=0.6)

        # Complex function with multiple minima
        def complex_function(x):
            return 0.4 * x**4 - 1.5 * x**2 + 0.3 * x + 1

        complex_graph = problem_axes.get_graph(
            complex_function,
            color=YELLOW,
            stroke_width=4,
            x_range=[-2.8, 2.8]
        )

        # Highlight local and global minima
        local_min_point = problem_axes.c2p(-0.8, complex_function(-0.8))
        global_min_point = problem_axes.c2p(1.2, complex_function(1.2))

        local_min_dot = GlowDot(local_min_point, color=RED, radius=0.08)
        global_min_dot = GlowDot(global_min_point, color=GREEN, radius=0.08)

        local_label = Text("Local Minimum", font_size=20, color=RED).next_to(local_min_dot, UP, buff=0.2)
        global_label = Text("Global Minimum", font_size=20, color=GREEN).next_to(global_min_dot, UP, buff=0.2)

        # Animate the problem setup
        self.play(Write(problem_text, run_time=0.8))
        self.play(Write(problem_desc, run_time=1))
        self.play(ShowCreation(problem_axes, run_time=1))
        self.play(ShowCreation(complex_graph, run_time=1.5))

        # Highlight the minima
        self.play(
            FadeIn(local_min_dot),
            FadeIn(global_min_dot),
            Write(local_label),
            Write(global_label),
            run_time=1
        )

        # Add a confused hiker animation
        hiker_start = problem_axes.c2p(-2.5, complex_function(-2.5))
        hiker = GlowDot(hiker_start, color=ORANGE, radius=0.06)

        self.play(FadeIn(hiker))

        # Animate hiker moving around confused
        confused_path = [
            problem_axes.c2p(-1.8, complex_function(-1.8)),
            problem_axes.c2p(-0.5, complex_function(-0.5)),
            problem_axes.c2p(0.2, complex_function(0.2)),
            problem_axes.c2p(-0.8, complex_function(-0.8)),  # Gets stuck in local minimum
        ]

        for point in confused_path:
            self.play(hiker.animate.move_to(point), run_time=0.8)

        confused_text = Text("How do we find the best path?", font_size=28, color=ORANGE)
        confused_text.to_edge(DOWN, buff=1)

        self.play(Write(confused_text, run_time=1))
        self.wait(1)

        # Transition continues to mathematics

        # Section 2: The Mathematics
        math_title = Text("The Mathematics", font_size=48, color=GREEN)
        math_title.to_edge(UP, buff=0.5)

        # Create an elegant formula presentation
        formula_box = Rectangle(width=6, height=1.2, color=GREEN_C, fill_color=GREEN_E, fill_opacity=0.1)
        formula = Text("θₜ₊₁ = θₜ - α∇f(θₜ)", font_size=48, color=WHITE)
        formula.move_to(formula_box.get_center())

        formula_group = VGroup(formula_box, formula)

        # Add mathematical elegance with surrounding elements
        gradient_symbol = Text("∇", font_size=60, color=BLUE_C)
        gradient_symbol.next_to(formula_box, LEFT, buff=0.8)

        equals_symbol = Text("=", font_size=48, color=YELLOW_C)
        equals_symbol.next_to(formula_box, RIGHT, buff=0.8)

        # Animate the formula reveal
        self.play(
            Write(math_title, run_time=1),
            ShowCreation(formula_box, run_time=1),
            Write(formula, run_time=1.5),
            run_time=2
        )

        self.play(
            Write(gradient_symbol, run_time=0.8),
            Write(equals_symbol, run_time=0.8)
        )

        # Detailed parameter explanation with visual aids
        param_title = Text("Understanding Each Component:", font_size=36, color=BLUE_C)
        param_title.next_to(formula_group, DOWN, buff=1.2)

        # Create parameter cards
        theta_card = Rectangle(width=4, height=1.5, color=BLUE_C, fill_color=BLUE_E, fill_opacity=0.1)
        theta_symbol = Text("θ", font_size=42, color=BLUE_C)
        theta_desc = Text("Parameters", font_size=24, color=WHITE)
        theta_detail = Text("(what we optimize)", font_size=18, color=BLUE_B)
        theta_content = VGroup(theta_symbol, theta_desc, theta_detail).arrange(DOWN, buff=0.1)
        theta_content.move_to(theta_card.get_center())

        alpha_card = Rectangle(width=4, height=1.5, color=GREEN_C, fill_color=GREEN_E, fill_opacity=0.1)
        alpha_symbol = Text("α", font_size=42, color=GREEN_C)
        alpha_desc = Text("Learning Rate", font_size=24, color=WHITE)
        alpha_detail = Text("(step size)", font_size=18, color=GREEN_B)
        alpha_content = VGroup(alpha_symbol, alpha_desc, alpha_detail).arrange(DOWN, buff=0.1)
        alpha_content.move_to(alpha_card.get_center())

        gradient_card = Rectangle(width=4, height=1.5, color=PURPLE_C, fill_color=PURPLE_E, fill_opacity=0.1)
        gradient_symbol_card = Text("∇f", font_size=42, color=PURPLE_C)
        gradient_desc = Text("Gradient", font_size=24, color=WHITE)
        gradient_detail = Text("(steepest ascent)", font_size=18, color=PURPLE_B)
        gradient_content = VGroup(gradient_symbol_card, gradient_desc, gradient_detail).arrange(DOWN, buff=0.1)
        gradient_content.move_to(gradient_card.get_center())

        # Arrange parameter cards
        param_cards = VGroup(theta_card, alpha_card, gradient_card)
        param_cards.arrange(RIGHT, buff=0.5)
        param_cards.next_to(param_title, DOWN, buff=0.8)

        # Animate parameter cards with staggered entrance
        self.play(Write(param_title, run_time=1))

        self.play(
            LaggedStart(
                ShowCreation(theta_card),
                ShowCreation(alpha_card),
                ShowCreation(gradient_card),
                lag_ratio=0.3
            ),
            run_time=1.5
        )

        self.play(
            LaggedStart(
                Write(theta_content),
                Write(alpha_content),
                Write(gradient_content),
                lag_ratio=0.3
            ),
            run_time=1.5
        )

        # Add arrows showing the flow
        flow_arrow1 = Arrow(theta_card.get_right(), alpha_card.get_left(), color=YELLOW_C, buff=0.1)
        flow_arrow2 = Arrow(alpha_card.get_right(), gradient_card.get_left(), color=YELLOW_C, buff=0.1)
        flow_arrow3 = Arrow(gradient_card.get_right(), formula_box.get_left(), color=YELLOW_C, buff=0.1)

        minus_symbol = Text("-", font_size=36, color=RED_C)
        minus_symbol.next_to(flow_arrow3, RIGHT, buff=0.2)

        self.play(
            LaggedStart(
                ShowCreation(flow_arrow1),
                ShowCreation(flow_arrow2),
                ShowCreation(flow_arrow3),
                Write(minus_symbol),
                lag_ratio=0.2
            ),
            run_time=2
        )

        self.wait(2)

        # Spectacular transition to visualization
        self.play(
            FadeOut(math_title),
            FadeOut(formula_group),
            FadeOut(gradient_symbol),
            FadeOut(equals_symbol),
            FadeOut(param_title),
            FadeOut(param_cards),
            FadeOut(flow_arrow1),
            FadeOut(flow_arrow2),
            FadeOut(flow_arrow3),
            FadeOut(minus_symbol),
            FadeOut(complex_graph),
            run_time=1.5
        )

        # Section 3: Visual Demonstration - Make it EPIC!
        viz_title = Text("Visual Demonstration", font_size=52, color=RED_C, weight=BOLD)
        viz_title.to_edge(UP, buff=0.3)

        subtitle_viz = Text("Watch Gradient Descent in Action", font_size=28, color=RED_B)
        subtitle_viz.next_to(viz_title, DOWN, buff=0.3)

        # Create coordinate system
        axes = Axes(
            x_range=[-3, 3, 0.5],
            y_range=[-1, 4, 0.5],
            axis_config={"color": WHITE},
        )

        # Define a quadratic function: f(x) = x²
        def quadratic(x):
            return x**2

        # Create the graph
        graph = axes.get_graph(quadratic, color=YELLOW, x_range=[-2.5, 2.5])

        # Simple labels using Text
        x_label = Text("x", font_size=24, color=WHITE).next_to(axes.x_axis, DOWN, buff=0.5)
        y_label = Text("f(x) = x²", font_size=24, color=WHITE).next_to(axes.y_axis, LEFT, buff=0.5).rotate(90 * DEGREES)

        # Starting point
        start_point = axes.c2p(2.2, quadratic(2.2))
        dot = GlowDot(start_point, color=RED, radius=0.15)

        # Gradient vector (tangent line)
        def get_tangent_line(x):
            slope = 2 * x  # derivative of x² is 2x
            y = quadratic(x)
            point = axes.c2p(x, y)
            direction = np.array([1, slope, 0])  # 3D direction vector
            direction = direction / np.linalg.norm(direction)
            return Line(
                point - 2 * direction,
                point + 2 * direction,
                color=GREEN,
                stroke_width=4
            )

        tangent = get_tangent_line(2.2)
        gradient_label = Text("Gradient", font_size=24, color=GREEN)
        gradient_label.next_to(tangent.get_center(), RIGHT + UP, buff=0.2)

        self.play(Write(viz_title))
        self.wait(0.5)
        self.play(ShowCreation(axes), Write(x_label), Write(y_label))
        self.play(ShowCreation(graph))
        self.wait(0.5)
        self.play(FadeIn(dot))
        self.wait(0.5)

        # Show gradient
        self.play(ShowCreation(tangent), Write(gradient_label))
        self.wait(1)

        # Animate gradient descent steps
        learning_rate = 0.3
        current_x = 2.2
        step_count = 0

        # Create step labels
        step_labels = VGroup()
        for i in range(5):
            step_text = Text(f"Step {i+1}", font_size=20, color=ORANGE)
            step_labels.add(step_text)

        step_labels.arrange(DOWN, buff=0.3).to_corner(UL, buff=0.5)

        # Animate the descent
        for i in range(5):
            # Calculate gradient (derivative)
            gradient = 2 * current_x

            # Update position
            new_x = current_x - learning_rate * gradient
            new_point = axes.c2p(new_x, quadratic(new_x))

            # Create new tangent line
            new_tangent = get_tangent_line(new_x)

            # Animate the step
            self.play(
                dot.animate.move_to(new_point),
                Transform(tangent, new_tangent),
                FadeIn(step_labels[i]),
                run_time=1.5
            )

            current_x = new_x

            if i < 4:  # Don't wait on last step
                self.wait(0.5)

        # Show convergence
        convergence_text = Text("Converging to minimum!", font_size=36, color=GREEN)
        convergence_text.next_to(axes, DOWN, buff=0.5)

        self.play(Write(convergence_text))
        self.wait(1)

        # Section 4: Learning Rate Importance
        self.play(FadeOut(viz_title), FadeOut(axes), FadeOut(graph), FadeOut(dot),
                 FadeOut(tangent), FadeOut(gradient_label), FadeOut(convergence_text),
                 FadeOut(step_labels))

        lr_title = Text("Learning Rate Matters", font_size=48, color=PURPLE)
        lr_title.to_edge(UP, buff=0.5)

        # Show different learning rates
        lr_examples = VGroup(
            Text("α = 0.01 (too small - slow convergence)", font_size=32, color=PURPLE),
            Text("α = 0.1 (good)", font_size=32, color=PURPLE),
            Text("α = 1.0 (too large - may oscillate or diverge)", font_size=32, color=PURPLE)
        )

        lr_examples.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        lr_examples.next_to(lr_title, DOWN, buff=1)

        self.play(Write(lr_title))
        self.wait(0.5)
        for example in lr_examples:
            self.play(Write(example))
            self.wait(0.8)

        self.wait(1)

        # Section 5: Applications
        self.play(FadeOut(lr_title), FadeOut(lr_examples))

        applications_title = Text("Applications", font_size=48, color=BLUE)
        applications_title.to_edge(UP, buff=0.5)

        # Use Text objects instead of BulletedList to avoid LaTeX
        app1 = Text("• Machine Learning model training", font_size=32, color=WHITE)
        app2 = Text("• Neural network optimization", font_size=32, color=WHITE)
        app3 = Text("• Linear regression fitting", font_size=32, color=WHITE)
        app4 = Text("• Any optimization problem!", font_size=32, color=WHITE)

        applications = VGroup(app1, app2, app3, app4)
        applications.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        applications.next_to(applications_title, DOWN, buff=0.8)

        self.play(Write(applications_title))
        self.wait(0.5)
        for item in applications:
            self.play(Write(item))
            self.wait(0.7)

        # Conclusion
        self.play(FadeOut(applications_title), FadeOut(applications))

        conclusion = Text("Gradient descent finds function minima by following the gradient downhill", font_size=36, color=YELLOW)
        conclusion.to_edge(UP, buff=1)

        final_formula = Text("θₜ₊₁ = θₜ - α∇f(θₜ)", font_size=48, color=WHITE)
        final_formula.next_to(conclusion, DOWN, buff=0.8)

        self.play(Write(conclusion))
        self.wait(1)
        self.play(Write(final_formula))
        self.wait(2)

        # Fade out
        self.play(FadeOut(conclusion), FadeOut(final_formula))


class EndScreen(PatreonEndScreen):
    pass
