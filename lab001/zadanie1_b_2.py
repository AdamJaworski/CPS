import matplotlib.pyplot as plot
from zadanie1 import add_to_plot

add_to_plot(230, 50, 10000, 1, 'b-' )
add_to_plot(230, 50, 26,    1, 'g-o')
add_to_plot(230, 50, 25,    1, 'r-o')
add_to_plot(230, 50, 24,    1, 'k-o')
plot.grid()
plot.show()

# to samo co w b) - błędne odwzorwanie funckji
