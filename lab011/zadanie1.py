import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Wstęp z intrukcji
samplerate, x = wavfile.read('DontWorryBeHappy.wav')
x = x.mean(axis=1)
a = 0.9545
x = np.array(x)
d = x - a * np.concatenate(([0], x[:-1]))


# Quantization function
def lab1_kwant(d):
    min_val = np.min(d)
    max_val = np.max(d)
    levels = 16 #32
    step_size = (max_val - min_val) / (levels - 1)

    quantized_d = np.round((d - min_val) / step_size) * step_size + min_val
    return quantized_d


dq = lab1_kwant(d)

# Dekoder
y = np.zeros_like(dq)
for i in range(1, len(dq)):
    y[i] = dq[i] + a * y[i - 1]

# Plot the results
plt.figure()
n = np.arange(len(x))
plt.plot(n, x, 'b', label='x(n)')
plt.plot(n, dq, 'k', label='d(n)')
plt.plot(n, y, 'r', label='y(n)')
plt.legend()
plt.show()