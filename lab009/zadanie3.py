import numpy as np
import matplotlib.pyplot as plt

# Parametry symulacji
fs = 12e3  # Częstotliwość próbkowania (100 kHz)
T = 5      # Czas trwania symulacji w sekundach
t = np.arange(0, T, 1/fs)  # Wektor czasu

# Parametry sygnału
f_pilot = 19000  # Częstotliwość sygnału pilota (19 kHz)
mod_index = 10   # Indeks modulacji dla ±10 Hz
mod_freq = 0.1   # Częstotliwość modulacji (0.1 Hz)

# Sygnał pilota z modulacją częstotliwości
pilot_signal = np.sin(2 * np.pi * (f_pilot + mod_index * np.sin(2 * np.pi * mod_freq * t)) * t)

# Implementacja PLL
freq = 2 * np.pi * f_pilot / fs
theta = np.zeros(len(t) + 1)
alpha = 1e-2
beta = alpha**2 / 4

for n in range(len(t)):
    error =  -t[n] * np.sin(theta[n])
    theta[n + 1] = theta[n] + freq + alpha * error
    freq += beta * error

# Usuwamy pierwszy element theta dla zgodności rozmiarów
theta = theta[1:]

# Wizualizacja sygnału pilota i PLL
plt.figure(figsize=(10, 6))
plt.plot(t, pilot_signal, label='Sygnał pilota')
plt.plot(t, np.sin(theta), label='Sygnał PLL', linestyle='--')
plt.title('Porównanie sygnału pilota i sygnału wygenerowanego przez PLL')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.show()
