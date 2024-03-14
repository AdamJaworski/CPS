import matplotlib.pyplot as plot
from zadanie1 import add_to_plot

add_to_plot(230, 50, 10000, 1, 'b-' )
add_to_plot(230, 50, 51,    1, 'g-o')
add_to_plot(230, 50, 50,    1, 'r-o')
add_to_plot(230, 50, 49,    1, 'k-o')
plot.grid()
plot.show()

# z racji tego że sin nie spełniają warunku próbkowania , są one odwzorowane w "przekłamany sposób"
