import matplotlib.pyplot as plot
from zadanie1 import add_to_plot

# for freq in range(5, 300, 5):
#     add_to_plot(230, freq, 100, 1, 'b-o', f'{freq}')
# ^ nie czytelne


# ############Aliasing################
# (----------------------------------)
# 5, 105, 205
# add_to_plot(230, 5,   100, 1, 'r-o')
# add_to_plot(230, 105, 100, 1, 'g-o')
# add_to_plot(230, 205, 100, 1, 'b-o')
# nic nie widać bo 105 i 205 > Fs
# (----------------------------------)
# 95, 195, 205
# add_to_plot(230, 95,  100, 1, 'r-o')
# add_to_plot(230, 195, 100, 1, 'g-o')
# add_to_plot(230, 205, 100, 1, 'b-o')
# Aliasing
# (----------------------------------)
# 95, 105 sin
# add_to_plot(230,  95, 100, 1, 'r-o')
# add_to_plot(230, 105, 100, 1, 'g-o')
# ????
# (----------------------------------)
# 95, 105 cos - zmiana w zadanie1.py
# add_to_plot(230,  95, 100, 1, 'r-o')
# add_to_plot(230, 105, 100, 1, 'g-o')
# ????
# (----------------------------------)


plot.grid()
plot.show()


