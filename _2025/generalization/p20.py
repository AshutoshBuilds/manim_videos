from manimlib import *
import os
import scipy.special

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
TEST_BLUE='#008080'  # Teal color for all test/blue elements

# graphics_dir = os.path.expanduser('~/Stephencwelch Dropbox/welch_labs/double_descent/graphics/')
# svg_dir = os.path.expanduser('~/Downloads')
graphics_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/double_descent/graphics/'
svg_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/double_descent/graphics/to_manim'


def fit_legendre_pinv(x_train, y_train, degree, x_min=-2, x_max=2):
    """Fit Legendre polynomial using pseudoinverse"""
    # Rescale x to [-1, 1]
    x_scaled = 2 * (x_train - x_min) / (x_max - x_min) - 1
    
    feature_degrees = np.arange(degree + 1)[:, None]
    X_train_poly = scipy.special.eval_legendre(feature_degrees, x_scaled).T
    beta_hat = np.linalg.pinv(X_train_poly) @ y_train
    return beta_hat

def eval_legendre_poly(beta, x, degree, x_min=-2, x_max=2):
    """Evaluate Legendre polynomial with given coefficients"""
    # Rescale x to [-1, 1]
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    
    feature_degrees = np.arange(degree + 1)[:, None]
    X_poly = scipy.special.eval_legendre(feature_degrees, x_scaled).T
    return X_poly @ beta

def get_noisy_data(n_points=10, noise_level=0.2, random_seed=428):
    np.random.seed(random_seed) 
    x=np.random.uniform(-2, 2, n_points)
    y=f(x)+noise_level*np.random.randn(n_points)
    return x,y

def f(x): return 0.5*(x**2)
# def f(x): return 0.5*(x**4-3*x**2)
# def f(x): return np.add(2.0 * x, np.cos(x * 25)) #[:, 0]


class p20(InteractiveScene):
    def construct(self):
        # Configuration Parameters
        random_seed = 428
        n_points = 10
        noise_level = 0.2
        
        # Load SVG Axes
        curve_fit_axis_svg = SVGMobject(svg_dir + '/p8_15_2a.svg')[1:] 
        curve_fit_axis_svg.scale(4.0)
        curve_fit_axis_svg.move_to([-2.86, 0.6, 0])
        
        # Generate Training and Testing Data
        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)
        
        n_train_points = int(np.floor(n_points * 0.5))
        n_test_points = n_points - n_train_points
        x, y = get_noisy_data(n_points, noise_level, random_seed)
        
        x_train, y_train = x[:n_train_points], y[:n_train_points]
        x_test, y_test = x[n_train_points:], y[n_train_points:]
        
        # Create First Axes (Curve Fitting Plot)
        axes_1 = Axes(
            x_range=[-2.0, 2.0, 1],
            y_range=[-0.5, 2.0, 1],
            width=6,
            height=5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": True,
                "include_numbers": True,
                "include_tip": True,
                "stroke_width": 3,
                "tip_config": {"width": 0.02, "length": 0.02}
            }
        )
        axes_1.move_to([-3, 0, 0])
        
        # Create Parabola and Data Points
        parabola = axes_1.get_graph(
            lambda x: f(x),
            x_range=[-2, 2],
            color=CHILL_BROWN
        )
        parabola.set_stroke(width=3)
        
        train_dots = VGroup(*[
            Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) 
            for i in range(len(x_train))
        ])
        test_dots = VGroup(*[
            Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) 
            for i in range(len(x_test))
        ])
        all_dots = VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)
        test_dots.set_color('#008080')
        
        # Sort dots by x-coordinate for smooth animation
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]

        legend = VGroup()
        
        # Training Data item
        legend_training_dot = Dot(radius=0.06).set_color(YELLOW)
        legend_training_text = Text("Training Data", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_training = VGroup(legend_training_dot, legend_training_text).arrange(RIGHT, buff=0.15, aligned_edge=ORIGIN)
        
        # Testing Data item
        legend_testing_dot = Dot(radius=0.06).set_color(TEST_BLUE)
        legend_testing_text = Text("Testing Data", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_testing = VGroup(legend_testing_dot, legend_testing_text).arrange(RIGHT, buff=0.15, aligned_edge=ORIGIN)
        
        # Target Function item
        legend_target_line = Line(LEFT * 0.2, RIGHT * 0.2, color=CHILL_BROWN, stroke_width=3)
        legend_target_text = Text("Target Function", font_size=20, font='myraid-pro').set_color(CHILL_BROWN)
        legend_target = VGroup(legend_target_line, legend_target_text).arrange(RIGHT, buff=0.15, aligned_edge=ORIGIN)
        
        # Arrange all items in the legend
        legend_items = VGroup(legend_training, legend_testing, legend_target).arrange(RIGHT, buff=0.4)
        
        # Create box around legend
        legend_box = RoundedRectangle(
            width=legend_items.get_width() + 0.5,
            height=legend_items.get_height() + 0.35,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box.set_stroke(opacity=0.7)
        legend_items.move_to(legend_box.get_center())
        
        legend.add(legend_box, legend_items)
        legend.scale(0.85)
        # Position under the first plot - x-aligned with axes_1 at x=-3
        legend.move_to([-3, -3.2, 0])
        

        self.frame.reorient(0, 0, 0, (-3.21, 0.36, 0.0), 7.45)
        self.add(curve_fit_axis_svg, parabola, all_dots)
        self.wait()


        # degree3 = 3
        # beta_hat3 = fit_legendre_pinv(x_train, y_train, degree3)
        # all_y_fit3 = eval_legendre_poly(beta_hat3, all_x, degree3)
        # y_train_pred3 = eval_legendre_poly(beta_hat3, x_train, degree3)
        # y_test_pred3 = eval_legendre_poly(beta_hat3, x_test, degree3)
        # train_error_3 = np.mean((y_train - y_train_pred3)**2)
        # test_error_3 = np.mean((y_test - y_test_pred3)**2)
        

        # fit_points_3 = [axes_1.c2p(all_x[i], all_y_fit3[i]) for i in range(len(all_x))]
        # fit_line_3 = VMobject(color=ORANGE, stroke_width=3)
        # fit_line_3.set_points_smoothly(fit_points_3)


        # self.add(fit_line_3)

        all_fits=np.load('/Users/stephen/Stephencwelch Dropbox/welch_labs/double_descent/graphics/regularization_fits_oct_15_1.npy')


        all_fit_lines=VGroup()
        for fit in all_fits:
            fit_points = [axes_1.c2p(all_x[i], fit[i]) for i in range(len(all_x))]
            fit_line = VMobject(color=ORANGE, stroke_width=3)
            fit_line.set_points_smoothly(fit_points)
            all_fit_lines.add(fit_line)

        lambdas=np.arange(0, 0.31, 0.01)

        # self.add(all_fit_lines[10])

        # Ok Claude, can you complete this section? 
        # I want to add a little Tex ojbect on screen that
        # prints out the current value of lambda as I bring in each curve
        # let's try reducing the opacity of the non-current curve and see
        # how that looks.  

        # for lambd, fit_line in zip(lambdas, all_fit_lines):

        # Create lambda display text with Tex
        lambda_display = Tex(r"\lambda = 0.00", font_size=36).set_color(ORANGE)
        # lambda_display.move_to([-3, -2.5, 0])  # Position in upper right area
        lambda_display.move_to([1.5, .5, 0])

        self.add(lambda_display)
        self.play(ShowCreation(all_fit_lines[0]), run_time=2)
        self.wait()

        # Animate through all fit lines
        for i, (lambd, fit_line) in enumerate(zip(lambdas, all_fit_lines)):
            # Update lambda display
            new_lambda_display = Tex(f"\\lambda = {lambd:.2f}", font_size=36).set_color(ORANGE)
            new_lambda_display.move_to([1.5, .5, 0])
            
            if i == 0:
                # First curve - just add it
                self.remove(all_fit_lines[0])
                self.add(fit_line)
                self.play(
                    Transform(lambda_display, new_lambda_display),
                    run_time=0.5
                )
            else:
                # Subsequent curves - fade out previous, fade in new
                prev_fit_line = all_fit_lines[i-1]
                self.play(
                    prev_fit_line.animate.set_stroke(opacity=0.12),
                    FadeIn(fit_line),
                    Transform(lambda_display, new_lambda_display),
                    run_time=0.3
                )
            
            # Brief pause to see each curve
            self.wait(0.1)

        # At the end, show all curves with reduced opacity and highlight the last one
        # self.play(
        #     *[all_fit_lines[i].animate.set_opacity(0.15) for i in range(len(all_fit_lines)-1)],
        #     all_fit_lines[-1].animate.set_opacity(1.0),
        #     run_time=1
        # )





        self.wait(20)
        self.embed()





