import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Parametry sygnału
fs = 10000  # częstotliwość próbkowania
fn = 50     # częstotliwość nośna
fm = 1      # częstotliwość modulująca
df = 5      # głębokość modulacji
T = 1       # czas trwania sygnału

# Tworzenie sygnału
t = np.arange(0, T, 1/fs)  # wektor czasu
modulating_signal = df * np.sin(2 * np.pi * fm * t)  # sygnał modulujący
carrier_frequency = fn + modulating_signal  # zmienna częstotliwość nośna
modulated_signal = np.sin(2 * np.pi * carrier_frequency * t)  # sygnał zmodulowany

# Wyświetlanie sygnałów
plt.figure(figsize=(12, 6))
plt.plot(t, modulated_signal, label='Sygnał zmodulowany')
plt.plot(t, modulating_signal, label='Sygnał modulujący', linestyle='--')
plt.title('Sygnał zmodulowany i modulujący')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.show()

# Próbkowanie
fs_sampled = 25  # nowa częstotliwość próbkowania
t_sampled = np.arange(0, T, 1/fs_sampled)  # wektor czasu dla próbkowanego sygnału
modulated_signal_sampled = np.sin(2 * np.pi * (fn + df * np.sin(2 * np.pi * fm * t_sampled)) * t_sampled)

# Porównanie sygnałów
plt.figure(figsize=(12, 6))
plt.plot(t, modulated_signal, label='Sygnał zmodulowany (analogowy)')
plt.stem(t_sampled, modulated_signal_sampled, 'r', markerfmt='ro', basefmt=" ", linefmt='r', label='Sygnał próbkowany (fs=25 Hz)')
plt.title('Porównanie sygnałów zmodulowanych')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.show()

# Błędy spowodowane próbkowaniem
error = np.interp(t_sampled, t, modulated_signal) - modulated_signal_sampled
plt.figure(figsize=(12, 6))
plt.stem(t_sampled, error, 'g', markerfmt='go', basefmt=" ", linefmt='g', label='Błąd próbkowania')
plt.title('Błąd próbkowania')
plt.xlabel('Czas [s]')
plt.ylabel('Wartość błędu')
plt.legend()
plt.grid(True)
plt.show()

# Widmo gęstości mocy
def spectrum(signal, fs):
    N = len(signal)
    freq = np.fft.fftfreq(N, 1/fs)
    signal_fft = fft(signal)
    power_spectrum = np.abs(signal_fft)**2 / N
    return freq, power_spectrum

freq_analog, power_spectrum_analog = spectrum(modulated_signal, fs)
freq_sampled, power_spectrum_sampled = spectrum(modulated_signal_sampled, fs_sampled)

plt.figure(figsize=(12, 6))
plt.plot(freq_analog, power_spectrum_analog, label='Przed próbkowaniem')
plt.plot(freq_sampled, power_spectrum_sampled, label='Po próbkowaniu', linestyle='--')
plt.title('Widma gęstości mocy sygnału')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Gęstość mocy')
plt.xlim([-100, 100])  # ograniczenie zakresu dla lepszej czytelności
plt.legend()
plt.grid(True)
plt.show()
