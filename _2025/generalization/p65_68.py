from manimlib import *
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
TEST_BLUE='#008080' 

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

def get_fit_line(axes, x_train, y_train, x_test, y_test, all_x, degree=1, color=GREEN):
    beta_hat = fit_legendre_pinv(x_train, y_train, degree)
    all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
    y_train_pred = eval_legendre_poly(beta_hat, x_train, degree)
    y_test_pred = eval_legendre_poly(beta_hat, x_test, degree)
    all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
    train_error = np.mean((y_train - y_train_pred)**2)
    test_error = np.mean((y_test - y_test_pred)**2)

    fit_points = [axes.c2p(all_x[i], all_y_fit[i]) for i in range(len(all_x))]
    fit_line = VMobject(stroke_width=3)
    fit_line.set_points_smoothly(fit_points)
    fit_line.set_color(color)
    return fit_line, test_error, train_error, y_train_pred, y_test_pred


def f(x): return 0.5*(x**2)


class p65_68(InteractiveScene):
    '''
    Ok, so the idea here is to potentially do a bit of a composite thing
    in the dark region next to the book - 
    '''
    def construct(self):



        # Ok, so i think first let's bring in a version of both plots, maybe with some nice animation. 
        # I think we keep the same side-by-side orientation we used earlier, and
        # if I want to arange these plots vertically, I can do that in premiere. 




        curve_fit_axis_svg=SVGMobject(svg_dir+'/p8_15_2a.svg')[1:] 
        curve_fit_axis_svg.scale(4.0)
        curve_fit_axis_svg.move_to([-2.86, 0.6, 0])

        random_seed=428
        n_points=10
        noise_level=0.2


        all_x = np.linspace(-2, 2, 128)
        all_y = f(all_x)

        n_train_points=int(np.floor(n_points*0.5))
        n_test_points=n_points-n_train_points
        x,y=get_noisy_data(n_points, noise_level, random_seed)
                           
        x_train, y_train=x[:n_train_points], y[:n_train_points]
        x_test, y_test=x[n_train_points:],y[n_train_points:]

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
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_1.move_to([-3, 0, 0])

        parabola = axes_1.get_graph(
            lambda x: f(x),
            x_range=[-2, 2],
            color=CHILL_BROWN
        )
        parabola.set_stroke(width=3)


        train_dots = VGroup(*[Dot(axes_1.c2p(x_train[i], y_train[i]), radius=0.08) for i in range(len(x_train))])
        test_dots = VGroup(*[ Dot(axes_1.c2p(x_test[i], y_test[i]), radius=0.08) for i in range(len(x_test))])
        all_dots=VGroup(test_dots, train_dots)
        all_dots.set_color(YELLOW)

        #Sorted dots so I can bring them in nicely. 
        dots_with_x = []
        for i, dot in enumerate(train_dots):
            dots_with_x.append((x_train[i], dot, 'train'))
        for i, dot in enumerate(test_dots):
            dots_with_x.append((x_test[i], dot, 'test'))
        dots_with_x.sort(key=lambda item: item[0])
        sorted_dots = [item[1] for item in dots_with_x]


        fit_line_1, test_error_1, train_error_1, y_train_pred_1, y_test_pred_1 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=1, color=GREEN)
        fit_line_2, test_error_2, train_error_2, y_train_pred_2, y_test_pred_2 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=2, color=YELLOW)
        fit_line_3, test_error_3, train_error_3, y_train_pred_3, y_test_pred_3 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=3, color=ORANGE)
        fit_line_4, test_error_4, train_error_4, y_train_pred_4, y_test_pred_4 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=4, color='#FF00FF')
        fit_line_5, test_error_5, train_error_5, y_train_pred_5, y_test_pred_5 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=5, color='#FFFFFF')
        fit_line_10, test_error_10, train_error_10, y_train_pred_10, y_test_pred_10 = get_fit_line(axes_1, x_train, y_train, x_test, y_test, all_x, degree=10, color='#be1e2d')

        axes_2 = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.2, 1],
            width=6,
            height=5,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": True,
                "include_numbers": True,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_2.move_to([4.0, 0.48, 0])


        degrees = [1, 2, 3, 4, 5, 10]
        train_errors = [0.221350, 0.031825, 0.000803, 0.000000, 0.0, 0.0]
        test_errors = [0.691636, 0.079449, 0.727023, 1.328535, 0.694120, 0.481703]

        train_error_dots = VGroup(*[Dot(axes_2.c2p(degrees[i], train_errors[i]), radius=0.08)
                                    for i in range(len(degrees))])
        test_error_dots = VGroup(*[Dot(axes_2.c2p(degrees[i], test_errors[i]), radius=0.08)
                                  for i in range(len(degrees))])
        train_error_dots.set_color(YELLOW)
        test_error_dots.set_color(CYAN)
        test_error_dots.set_opacity(0.7)


        error_axis_svg=SVGMobject(svg_dir+'/p8_15_2-05.svg') #[1:] 
        degree_label=error_axis_svg[39:]
        error_axis_svg.scale(2.95)
        error_axis_svg.move_to([4.79, 0.75, 0])

        extended_axis_labels_svg=SVGMobject(svg_dir+'/p46_56_2-04.svg')[1:]
        extended_axis_svg=SVGMobject(svg_dir+'/p46_56_2-05.svg')[1:]
        extended_axis_group=Group(extended_axis_svg, extended_axis_labels_svg)
        extended_axis_group.scale(4.5)
        extended_axis_group.move_to([6.57, -1.98, 0])

        #Probably need to noodle with scale here!
        extended_axis_svg.scale([0.85, 1, 1], about_point=extended_axis_svg.get_left()) 
        
        test_dots.set_color('#008080')
        self.frame.reorient(0, 0, 0, (1.32, 0.56, 0.0), 9.80)

        parabola.set_stroke(opacity=0.5)
        self.wait()
        self.play(Write(curve_fit_axis_svg))
        self.play(ShowCreation(parabola), LaggedStart(*[FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        

        degree_label.scale(1.6)
        degree_label.move_to([5.5, -2.5, 0])
        self.wait()

        self.play(Write(error_axis_svg[1:]), 
            Write(extended_axis_svg), 
            Write(extended_axis_group[1][6]),
            Write(extended_axis_group[1][8]))

        double_descent_curve_svg=SVGMobject(svg_dir+'/p46_56_2-12.svg')

        double_descent_curve_svg.scale(2.6)
        double_descent_curve_svg.move_to([5.62, 1.12, 0])

        

        self.wait()
        self.play(ShowCreation(train_error_dots), ShowCreation(test_error_dots), run_time=2)
        self.play(Write(double_descent_curve_svg))
        self.wait()
        #Hmm kinda feel like i want the double descent curve in manim?


        # self.play(ShowCreation(double_descent_curve_svg))

        interp_threshold_line = DashedLine(
            start=axes_2.c2p(4, 0.0),
            end=axes_2.c2p(4, 1.5),
            color=WHITE,
            stroke_width=3,
            dash_length=0.1
        )

        interp_threshold_label = Text(
            "Interpolation Threshold",
            font_size=24,
            font='myraid-pro',
            color=WHITE
        )
        interp_threshold_label.next_to(axes_2.c2p(4, 1.5), UP, buff=0.2)

        self.wait()
        self.play(FadeIn(interp_threshold_line), Write(interp_threshold_label))

        strikethrough_line = Line(
            start=degree_label.get_left() + LEFT * 0.1,
            end=degree_label.get_right() + RIGHT * 0.1,
            color=YELLOW,
            stroke_width=3
        )
        flexibility_label = Text(
            "Flexibility",
            font_size=32,
            font='myriad-pro',
        ).set_color(YELLOW)

        flexibility_label.next_to(degree_label, DOWN, buff=0.1).shift([0.02, 0, 0])


        self.wait()
        self.play(ShowCreation(strikethrough_line), run_time=1.0)
        self.play(Write(flexibility_label), run_time=1)
        self.wait()









        self.wait(20)
        self.embed()




