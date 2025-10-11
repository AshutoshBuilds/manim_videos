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


class p8_15_1(InteractiveScene):
    '''
    
    '''
    def construct(self):

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


        # Create legend items
        legend_train_dot = Dot(radius=0.05).set_color(YELLOW)
        legend_train_text = Text("Training Data", font_size=18, font='myraid-pro').set_color(CHILL_BROWN)
        legend_train = VGroup(legend_train_dot, legend_train_text).arrange(RIGHT, buff=0.15)
        
        legend_test_dot = Dot(radius=0.05).set_opacity(0.5).set_color(BLUE)
        legend_test_text = Text("Testing Data", font_size=18, font='myraid-pro').set_color(CHILL_BROWN)
        legend_test = VGroup(legend_test_dot, legend_test_text).arrange(RIGHT, buff=0.15)
        
        legend_line = Line(LEFT * 0.2, RIGHT * 0.2, color=GREEN, stroke_width=3)
        legend_line_text = Tex("y=ax+b", font_size=23).set_color(GREEN)
        legend_line_item = VGroup(legend_line, legend_line_text).arrange(RIGHT, buff=0.15)
        
        # Arrange legend items horizontally
        legend_items = VGroup(legend_train, legend_test, legend_line_item).arrange(RIGHT, buff=0.3)
        
        # Create rounded rectangle background
        legend_box = RoundedRectangle(
            width=legend_items.get_width() + 0.6,
            height=legend_items.get_height() + 0.3,
            corner_radius=0.08,
            stroke_color=CHILL_BROWN,
            stroke_width=2,
            fill_color=None,
            fill_opacity=0.0
        )
        legend_box.set_stroke(opacity=0.7)
        
        # Position legend below the plot
        legend = VGroup(legend_box, legend_items)
        legend.move_to(axes_1.get_bottom() + DOWN * 0.15 + RIGHT * 0.1)



        self.frame.reorient(0, 0, 0, (-2.94, 0.08, 0.0), 7.45)

        self.play(Write(curve_fit_axis_svg), run_time=3)
        # self.play(ShowCreation(parabola), run_time=2.5)
        # self.play(ShowCreation(all_dots))
        self.play(ShowCreation(parabola), LaggedStart(*[FadeIn(dot) for dot in sorted_dots], lag_ratio=0.15), run_time=2)
        self.wait()

        self.play(test_dots.animate.set_color(CYAN).set_opacity(0.5))
        self.wait()

        #Need all of this or nah?
        degree=1
        beta_hat = fit_legendre_pinv(x_train, y_train, degree)
        all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
        y_train_pred = eval_legendre_poly(beta_hat, x_train, degree)
        y_test_pred = eval_legendre_poly(beta_hat, x_test, degree)
        all_y_fit = eval_legendre_poly(beta_hat, all_x, degree)
        train_error_1 = np.mean((y_train - y_train_pred)**2)
        test_error_1 = np.mean((y_test - y_test_pred)**2)

        fit_points = [axes_1.c2p(all_x[i], all_y_fit[i]) for i in range(len(all_x))]
        fit_line = VMobject(color=GREEN, stroke_width=3)
        fit_line.set_points_smoothly(fit_points)

        self.play(ShowCreation(fit_line), run_time=1.5)
        self.add(legend)

        self.wait()

        # Alright nice start here, now this is going to get a little wonky, might need to 
        # tweak or slow down the script a little - pauses are ok too
        # Anyway I want to bring a second error vs degree plot, and
        # i'd really like to draw little vertical lines between the fits and the points, 
        # and then move them over to show the height of points on the the plot kinda deal, ya know?
        # Here's the numbers I'll be plotting -> 
        # Degree   1: Train MSE = 0.221350, Test MSE = 0.691636
        # Degree   2: Train MSE = 0.031825, Test MSE = 0.079449
        # Degree   3: Train MSE = 0.000803, Test MSE = 0.727023
        # Degree   4: Train MSE = 0.000000, Test MSE = 1.328535
        # Degree   5: Train MSE = 0.000000, Test MSE = 0.694120
        # Degree  10: Train MSE = 0.000000, Test MSE = 0.481703
        # I can hard code them or compute them, maybe compute? No super strong opinion. 
        # For this opening scene, were' going to scatter plot degees 1-4, then add some curves on top. 
        # yeah let me go ahead and make an axis, and get all 8 dots on there and see how if feels
        # Then I'll repace the manim axis with a nice illustrator one



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


        degrees = [1, 2, 3, 4]
        train_errors = [0.221350, 0.031825, 0.000803, 0.000000]
        test_errors = [0.691636, 0.079449, 0.727023, 1.328535]

        train_error_dots = VGroup(*[Dot(axes_2.c2p(degrees[i], train_errors[i]), radius=0.08)
                                    for i in range(len(degrees))])
        test_error_dots = VGroup(*[Dot(axes_2.c2p(degrees[i], test_errors[i]), radius=0.08)
                                  for i in range(len(degrees))])
        train_error_dots.set_color(YELLOW)
        test_error_dots.set_color(CYAN)
        test_error_dots.set_opacity(0.7)


        error_axis_svg=SVGMobject(svg_dir+'/p8_15_2-05.svg') #[1:] 
        error_axis_svg.scale(2.95)
        error_axis_svg.move_to([4.79, 0.75, 0])


        # Ok this is going to get a little complicated, but should help make things really clear
        # I want to draw little yelllow vertical bars from the fit line to each training point, 
        # And then move the 5 vertical bars over to the error plot, scaling and aligning them to fit between the axis (0) 
        # the first training error point. Then I add the actually dot!
        # Not sure yet if I want to do this animation all at once when I bring in/create the new axis, 
        # or in phases. Let's try all at once first and then I can break apart if that doesn't work. 
        #Claude can create these lines and animation? Might make sense to use a replacment transform. 


        train_error_bars = VGroup()
        for i in range(len(x_train)):
            point_pos = axes_1.c2p(x_train[i], y_train[i])
            fit_pos = axes_1.c2p(x_train[i], y_train_pred[i])
            # Always draw from the lower point to the higher point
            if point_pos[1] > fit_pos[1]:  # point is above fit
                error_bar = Line(fit_pos, point_pos, color=YELLOW, stroke_width=3)
            else:  # point is below fit
                error_bar = Line(point_pos, fit_pos, color=YELLOW, stroke_width=3)
            train_error_bars.add(error_bar)

        # self.add(train_error_bars)

        target_bars = VGroup()
        bar_height = train_errors[0] / len(train_error_bars)  # Each bar gets equal portion of total height
        x_pos = degrees[0]  # x position is degree 1

        for i in range(len(train_error_bars)):
            # Stack bars on top of each other
            bottom_y = i * bar_height
            top_y = (i + 1) * bar_height
            bottom = axes_2.c2p(x_pos, bottom_y)
            top = axes_2.c2p(x_pos, top_y)
            target_bar = Line(bottom, top, color=YELLOW, stroke_width=3)
            target_bars.add(target_bar)

        self.wait()
        self.play(LaggedStart(*[ShowCreation(bar) for bar in train_error_bars], lag_ratio=0.1), run_time=1.5)

        train_error_bars_copy=train_error_bars.copy()
        self.add(train_error_bars_copy)
        train_error_bars.set_opacity(0.5)
        self.wait()
        self.play(
            self.frame.animate.reorient(0, 0, 0, (0.82, 0.46, 0.0), 8.86),
            legend.animate.shift([0, -0.1, 0]),
            Write(error_axis_svg),
            ReplacementTransform(train_error_bars, target_bars), #Too much visually all at once
            run_time=4
        )
        self.play(ShowCreation(train_error_dots[0]))

        self.wait()
        # self.play(ReplacementTransform(train_error_bars_copy, target_bars), run_time=3.0)
        
        self.wait()






        # self.play(self.frame.animate.reorient(0, 0, 0, (0.82, 0.46, 0.0), 8.86), 
        #           legend.animate.shift([0, -0.1, 0]), run_time=3)


        # # self.add(axes_2)
        # self.add(error_axis_svg)


        # self.add(train_error_dots)
        # self.add(test_error_dots)


        # train_dots.set_color(YELLOW)
        # test_dots.set_color(BLUE)
        # test_dots.set_opacity(0.5)
        # # parabola.set_stroke(opacity=0.5)


        # self.add(parabola, train_dots, test_dots)
        # self.add(curve_fit_axis_svg)

        # self.frame.reorient(0, 0, 0, (0.14, 0.38, 0.0), 8.31)


        error_curves_svg=SVGMobject(svg_dir+'/p8_15_2-06.svg') #[1:] 
        error_curves_svg.scale(2.98)
        error_curves_svg.move_to([4.84, 0.74, 0])



        self.embed()
        self.wait(20)


























