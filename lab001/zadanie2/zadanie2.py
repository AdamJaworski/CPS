import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# np.convolve !!!

T = 1
fs1 = 10000
fs3 = 200
frequency = 5

# Polecenie jest takie cieżkie do zrozumienia (który syngał ma jakie próbkowanie, częstoliwość amplitudę itd.) że dokonałem swoich założeń na podstawie
# zdjęcia - sygnał pseudno analogowy ma mniejsze próbkowanie niż sin(x)/x [sinc]

t_10000 = np.arange(0, T, 1/1000)
t_200   = np.arange(0, T, 1/200)

sinc_arg = 2 * t_10000 * frequency

sinc = np.sinc(sinc_arg)
sin_1 = np.sin(2 * np.pi * t_200 * frequency)

# syngał wyjściowy jest bardzo dokładny więc zakładam że jest tak samo spróbkowany jak sinc
xhat = np.zeros(len(t_10000))


# zrozumiałem że pracuje na zakresach wyzanczonych przez mniejsze próbkowanie
# zgodnie z tym założeniem muszę policzyć ilość iteracji na jedym zakresie (kiedy wartości się zaównają)
number_of_iterations = 5 # 1000 / 200

for index, t in enumerate(t_200):
    if index == 0 or index == 1 or index > 197:
        continue
    for i in range(number_of_iterations):
        # 5 punktów referencji, int() * 200 bo to jest array więc potrzebuję całkowity index
        point_1 = sin_1[int((t - 2 * 1/200) * 200)] * sinc[int(((t - 2 * 1/200) + (i * 1/1000)) * 1000)]
        point_2 = sin_1[int((t - 1 * 1/200) * 200)] * sinc[int(((t - 1 * 1/200) + (i * 1/1000)) * 1000)]
        point_3 = sin_1[int(t * 200)]               * sinc[int((t + (i * 1/1000)) * 1000)]
        point_4 = sin_1[int((t + 1 * 1/200) * 200)] * sinc[int(((t + 1 * 1/200) + (i * 1/1000)) * 1000)]
        point_5 = sin_1[int((t + 2 * 1/200) * 200)] * sinc[int(((t + 2 * 1/200) + (i * 1/1000)) * 1000)]
        xhat[int((t + (i * 1/1000)) * 1000)] += (point_1 + point_2 + point_3 + point_4 + point_5)

plt.plot(t_10000, xhat, 'r')
plt.plot(t_200 , sin_1, 'b')
plt.plot(t_10000, sinc, 'g')

plt.show()