import numpy as np
import matplotlib.pyplot as plt


def add_awgn_noise(signal, noise_power):
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    out = signal + noise
    return out


fs = 8000
t = 1
A1, f1 = -0.5,  34.2
A2, f2 =    1,  115.5
time = np.linspace(0, t, fs * t)

noise_lvls = [10, 20, 40]
signals = [A1 * np.sin(2 * np.pi * f1 * time) + A2 * np.sin(2 * np.pi * f2 * time)]

for power in noise_lvls:
    signals.append(add_awgn_noise(signals[0], power))

plt.figure(figsize=(14, 8))
lvls = [0, *noise_lvls]

for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, signals[i])
    plt.title(f'Sygnał z szumem o mocy {lvls[i]} dB')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
plt.tight_layout()
plt.show()

d_all = signals
filtered_signals = []
errors = []
M = 80
mi = 0.00000329

for d in d_all:
    x = np.append(d[0], d[:-1])

    y, e = np.zeros(len(x)), np.zeros(len(x))
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
    # print(e)
    plt.subplot(2, 1, 1)
    plt.plot(time, d)
    plt.title(f'Sygnał')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.subplot(2, 1, 2)
    plt.plot(time, y)
    plt.title(f'Zfiltrowany')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.tight_layout()
    plt.show()
