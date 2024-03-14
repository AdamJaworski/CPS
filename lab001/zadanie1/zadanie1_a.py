import matplotlib.pyplot as plot
from zadanie1 import add_to_plot

add_to_plot(230, 50, 10000, 0.1, 'b-' )
add_to_plot(230, 50, 500,   0.1, 'r-o')
add_to_plot(230, 50, 200,   0.1, 'k-x')
plot.grid()
plot.show()

# wszystkie sinusoidy spełniają kryterium  Fmax * 2 < Fs
# im wyższe Fs tym większa rozdzelczość funckji
