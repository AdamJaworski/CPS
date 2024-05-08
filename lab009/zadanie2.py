import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sounddevice as sd

fs, d = wav.read('mowa_3.wav')
fs2, x = wav.read('mowa_2.wav')
fs3, correct = wav.read('mowa_1.wav')

x = x / np.max(np.abs(x))
d = d / np.max(np.abs(d))
correct = correct / np.max(np.abs(correct))

M = 83      # dla 80 fajnie wykres wygląda
mi = 0.1

y = e = np.zeros(len(x))
bx = h = np.zeros(M)

for n in range(len(x)):
    bx = np.roll(bx, -1)
    bx[-1] = x[n]
    y[n] = np.dot(h, bx)
    e[n] = d[n] - y[n]
    if not np.any(np.isnan(e[n] * bx)):
        h += mi * e[n] * bx

signal_power = np.mean(d ** 2)
noise_power = np.mean((y - d) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"SNR: {snr_db} dB")

sd.play(y, fs)

plt.subplot(3, 1, 1)
plt.plot([*range(len(d))], correct)
plt.title(f'SA')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.subplot(3, 1, 2)
plt.plot([*range(len(d))], d)
plt.title(f'SA+G(SB)')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.subplot(3, 1, 3)
plt.plot([*range(len(y))], y)
plt.title(f'Zfiltrowany')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.tight_layout()
plt.show()

