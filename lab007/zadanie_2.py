import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, iirfilter

# Parametry sygnału FM i filtrów
fpr = 120e3  # częstotliwość próbkowania dla celów symulacji, wyższa niż rzeczywista
fc_pilot = 19e3  # częstotliwość pilota stereo (19 kHz)
f1_mono = 30  # dolna granica pasma audio mono
f2_mono = 15e3  # górna granica pasma audio mono
N_fir = 151  # ilość próbek dla filtru FIR, nieparzysta dla symetrii względem próbki środkowej

# FIR dla mon
fir_taps = firwin(N_fir, [f1_mono, f2_mono], pass_zero=False, window='hann', fs=fpr)

# FIR dla pilota
bandwidth = 200
N_fir_pilot = 200
f1_pilot = fc_pilot - bandwidth/2
f2_pilot = fc_pilot + bandwidth/2
fir_pilot_taps = firwin(N_fir_pilot, [f1_pilot, f2_pilot], pass_zero=False, window='hamming', fs=fpr)

# Charakterystyki częstotliwościowe dla filtru FIR
w_fir, h_fir = freqz(fir_taps, worN=8000, fs=fpr)
# Charakterystyki częstotliwościowe dla filtru IIR
w_pilot_fir, h_pilot_fir = freqz(fir_pilot_taps, worN=8000, fs=fpr)

# Rysowanie charakterystyk
plt.figure(figsize=(12, 6))

# Charakterystyka amplitudowa filtru FIR
plt.subplot(2, 2, 1)
plt.plot(w_fir, 20 * np.log10(abs(h_fir) + 1e-9))
plt.title('FIR Filter Amplitude Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()

# Charakterystyka fazowa filtru FIR
plt.subplot(2, 2, 2)
plt.plot(w_fir, np.unwrap(np.angle(h_fir)))
plt.title('FIR Filter Phase Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.grid()

# Charakterystyka amplitudowa filtru IIR
plt.subplot(2, 2, 3)
plt.plot(w_pilot_fir, 20 * np.log10(abs(h_pilot_fir) + 1e-9), label="FIR BP Filter")
plt.axvline(x=fc_pilot, color='red', linestyle='--', label="19 kHz Pilot")
plt.title('FIR BP Filter Amplitude Response (for 19 kHz Pilot)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.legend()
plt.grid()

# Charakterystyka fazowa filtru IIR
plt.subplot(2, 2, 4)
plt.plot(w_pilot_fir, np.unwrap(np.angle(h_pilot_fir)), label="FIR BP Filter")
plt.title('FIR BP Filter Phase Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

t = np.linspace(0, 1, int(fpr/ 200), endpoint=False)
pilot = np.sin(2 * np.pi * fc_pilot * t)

filtered_signal = np.convolve(pilot, fir_pilot_taps, mode='same')
frequencies = np.fft.fftfreq(len(t), 1/fpr)
spectrum = np.fft.fft(filtered_signal)
spectrum_pilot = np.fft.fft(pilot)

plt.figure(figsize=(14, 6))

# Sygnał oryginalny i przefiltrowany w domenie czasu
plt.subplot(2, 1, 1)
plt.plot(t, pilot, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal', alpha=0.75)
plt.title('Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Widmo sygnału oryginalnego i przefiltrowanego w domenie częstotliwości
plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies) // 2], 20 * np.log10(np.abs(spectrum_pilot[:len(spectrum_pilot) // 2]) + 1e-9), label='Original Signal')
plt.plot(frequencies[:len(frequencies) // 2], 20 * np.log10(np.abs(spectrum[:len(spectrum) // 2]) + 1e-9), label='Filtered Signal', alpha=0.75)
plt.title('Signal Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.xlim(fc_pilot - 1000, fc_pilot + 1000)
plt.legend()

plt.tight_layout()
plt.show()