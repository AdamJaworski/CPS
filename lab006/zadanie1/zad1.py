from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Dane filtru analogowego
z_analog = np.array([])
p_analog = np.array([ -48.82441607+7712.16475812j,  -48.82441607-7712.16475812j,
                      -116.83326605+7642.68591281j, -116.83326605-7642.68591281j,
                      -47.35442071+7479.96850595j,  -47.35442071-7479.96850595j,
                      -115.36298613+7546.50707603j, -115.36298613-7546.50707603j])
k_analog = 3989876368.752743

# Częstotliwość próbkowania
fs = 16000  # Hz

# Konwersja filtru analogowego na cyfrowy za pomocą transformaty biliniowej
b_digital, a_digital = signal.bilinear(*signal.zpk2tf(z_analog, p_analog, k_analog), fs)

# Obliczanie charakterystyki amplitudowo-częstotliwościowej dla filtru analogowego
w_analog, h_analog = signal.freqs_zpk(z_analog, p_analog, k_analog, worN=np.logspace(np.log10(100), np.log10(10000), 500))

# Obliczanie charakterystyki amplitudowo-częstotliwościowej dla filtru cyfrowego
w_digital, h_digital = signal.freqz(b_digital, a_digital, worN=np.logspace(np.log10(100), np.log10(10000), 500), fs=fs)

# Rysowanie charakterystyki amplitudowo-częstotliwościowej
plt.figure(figsize=(14, 7))
plt.semilogx(w_analog, 20 * np.log10(abs(h_analog)), label='Filtr analogowy')
plt.semilogx(w_digital, 20 * np.log10(abs(h_digital)), label='Filtr cyfrowy', linestyle='--')
plt.axvline(1189, color='r', linestyle=':', label='Dolna częstotliwość graniczna')
plt.axvline(1229, color='g', linestyle=':', label='Górna częstotliwość graniczna')
plt.title('Porównanie charakterystyk amplitudowo-częstotliwościowych')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.grid(True, which='both', axis='both')
plt.legend()
plt.show()

fs = 16000  # częstotliwość próbkowania
T = 1       # czas trwania sygnału w sekundach
t = np.arange(0, T, 1/fs)  # wektor czasu

# Generowanie sygnału
f1 = 1209  # częstotliwość pierwszej harmonicznej
f2 = 1272  # częstotliwość drugiej harmonicznej
signal_input = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

plt.figure(figsize=(14, 7))
plt.plot(t, signal_input)
plt.title('Sygnał wejściowy')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()


def manual_filter(x, b, a):
    N = len(x)
    M = len(b)
    L = len(a)
    y = np.zeros(N)

    for n in range(N):
        # Suma dla licznika
        for k in range(M):
            if n - k >= 0:  # Sprawdzenie, czy nie wychodzimy poza zakres
                y[n] += b[k] * x[n - k]

        # Suma dla mianownika (pomiń a[0], ponieważ zakładamy, że a[0] = 1)
        for l in range(1, L):
            if n - l >= 0:  # Sprawdzenie, czy nie wychodzimy poza zakres
                y[n] -= a[l] * y[n - l]

    return y


y = manual_filter(signal_input, b_digital, a_digital)

plt.figure(figsize=(14, 7))
plt.plot(t, y)
plt.title('Sygnał wejściowy')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()
