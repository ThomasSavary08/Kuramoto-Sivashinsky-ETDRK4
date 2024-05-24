# Libraries
import argparse
import numpy as np
from manim import *
from coefficients import *
from manim.utils.file_ops import open_file as open_media_file
from functions import drawable_functions, drawable_functions_names

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type = str, required = True, help = "Name of the function to draw.", default = "Fish")
parser.add_argument('-n', '--number', type = int, required = True, help = "Number of coefficients for Fourier series", default = 301)

# Read arguments
args = parser.parse_args()
input_, N = args.input, args.number
K = int(N//2)

# Analyse arguments
if input_ in drawable_functions_names:
    i = 0
    while i < len(drawable_functions):
        if drawable_functions[i].name == input_:
            coeffs = np.array(drawable_functions[i].get_coefficients(K, ordered = True))
            break
        else:
            i += 1 
else:
    print("The input is not part of drawable functions, but you can always add a function in the functions.py file.")
    exit()

# Drawing with Fourier series
class FourierDrawing(Scene):
    
    def __init__(self):
        super().__init__()
        self.K = K
        self.N = N
        self.T = drawable_functions[i].T
        self.coeffs = coeffs
        self.vector_config = {
            "buff": 0,
            "max_tip_length_to_length_ratio": 0.25,
            "tip_length": 0.15,
            "max_stroke_width_to_length_ratio": 10,
            "stroke_width": 1.4
        }
        self.circle_config = {
            "stroke_width": 1,
            "stroke_opacity": 0.3,
            "color": WHITE
        }
        self.path_config = {
            "stroke_color": YELLOW,
            "stroke_width": 2,
            "stroke_opacity": 1
            }
        self.clock = ValueTracker()
        
    def get_vectors(self):
        
        # Instanciate a group of vectors
        vectors = VGroup()

        # Add first vector
        x, y = np.real(self.coeffs[0]), np.imag(self.coeffs[0])
        vector = Vector(np.array([x,y]), **self.vector_config)
        vector.k = 0
        vector.x = x
        vector.y = y
        vector.coeff = self.coeffs[0]
        vectors.add(vector) 
            
        # Add others
        for i in range(1, len(self.coeffs)):

            # Compute k (c_k)
            if (i%2 == 0):
                k = int(i/2)
            else:
                k = -int((i+1)/2)
            
            # Instanciate and add new vector
            start = np.array([vectors[-1].x, vectors[-1].y, 0.])
            x, y = np.real(self.coeffs[i]), np.imag(self.coeffs[i])
            direction = np.array([x,y])
            vector = Vector(direction, **self.vector_config).shift(start)
            vector.k = k
            vector.x = vectors[-1].x + x
            vector.y = vectors[-1].y + y
            vector.coeff = self.coeffs[i]
            vectors.add(vector)
        
        return vectors

    def update_vectors(self, vectors: VGroup):

        # Pulsation
        t = self.clock.get_value()
        w = (2.*np.pi)/self.T

        # Update vectors
        for i in range(1, len(vectors)):
            start = vectors[i-1].get_end()
            point = vectors[i].coeff * np.exp(1.j*w*vectors[i].k*t)
            end = start + np.array([np.real(point), np.imag(point), 0.])
            vectors[i].put_start_and_end_on(start, end)

    def get_circles(self, vectors: VGroup):
            
        # Instanciate a group of circles
        circles = VGroup()

        # Instanciate each circle and add it
        for vector in vectors:
            radius = np.linalg.norm(vector.get_end() - vector.get_start())
            circle = Circle(radius, **self.circle_config)
            circle.move_to(vector.get_start())
            circle.vector = vector
            circles.add(circle)
        
        return circles

    def update_circles(self, circles: VGroup):
        
        # Move the center of each circle
        for i in range(2, len(circles)):
            circles[i].move_to(circles[i].vector.get_start())

    def get_path(self, vectors: VGroup):
        
        # Instanciate path
        path = TracedPath(vectors[-1].get_end, **self.path_config)
        return path

    def construct(self):

        # Instanciate vectors, circles and path
        vectors = self.get_vectors()
        circles = self.get_circles(vectors)
        drawn_path = self.get_path(vectors)

        # Draw initial vectors and circles
        self.wait(1)
        self.play(*[GrowArrow(arrow) for arrow in vectors], *[Create(circle) for circle in circles], run_time = 2.5)
        self.add(vectors, circles, drawn_path)

        # Run animation
        vectors.add_updater(self.update_vectors)
        circles.add_updater(self.update_circles)
        self.play(self.clock.animate.set_value(2*self.T), run_time = 20)
        self.wait(2)

# Launch of the video animation
if __name__ == '__main__':
    scene = FourierDrawing()
    scene.render()
    open_media_file(scene.renderer.file_writer.movie_file_path)