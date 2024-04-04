from scipy.signal import buttord, cheb1ord, cheb2ord, ellipord, butter, cheby1, cheby2, ellip, freqs
import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency and cutoff frequencies
f_s = 256e3  # Sampling frequency 256kHz
f_c = f_s / 2  # Nyquist frequency

# Passband and stopband specifications
f_pass = f_s / 64  # Passband cutoff frequency
f_stop = f_s / 2   # Stopband cutoff frequency
ripple_pass = 3    # Passband ripple in dB
attenuation_stop = 40  # Stopband attenuation in dB

# Normalize frequencies by the Nyquist frequency
Wp = f_pass / f_c
Ws = f_stop / f_c

# Calculate the minimum order for Butterworth filter
N_butt, Wn_butt = buttord(Wp, Ws, ripple_pass, attenuation_stop, analog=True)

# Calculate the minimum order for Chebyshev Type I filter
N_cheby1, Wn_cheby1 = cheb1ord(Wp, Ws, ripple_pass, attenuation_stop, analog=True)

# Calculate the minimum order for Chebyshev Type II filter
N_cheby2, Wn_cheby2 = cheb2ord(Wp, Ws, ripple_pass, attenuation_stop, analog=True)

# Calculate the minimum order for Elliptical filter
N_ellip, Wn_ellip = ellipord(Wp, Ws, ripple_pass, attenuation_stop, analog=True)

# Create the filters
b_butt, a_butt = butter(N_butt, Wn_butt, btype='low', analog=True)
b_cheby1, a_cheby1 = cheby1(N_cheby1, Wn_cheby1, ripple_pass, btype='low', analog=True)
b_cheby2, a_cheby2 = cheby2(N_cheby2, Wn_cheby2, ripple_pass, btype='low', analog=True)
b_ellip, a_ellip = ellip(N_ellip, ripple_pass, attenuation_stop, Wn_ellip, btype='low', analog=True)

# Frequency range for plotting
freqs_range = np.logspace(1, np.log10(f_s), 1000)

# Calculate frequency response for each filter
w_butt, h_butt = freqs(b_butt, a_butt, worN=freqs_range)
w_cheby1, h_cheby1 = freqs(b_cheby1, a_cheby1, worN=freqs_range)
w_cheby2, h_cheby2 = freqs(b_cheby2, a_cheby2, worN=freqs_range)
w_ellip, h_ellip = freqs(b_ellip, a_ellip, worN=freqs_range)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.semilogx(w_butt, 20 * np.log10(abs(h_butt)), label='Butterworth')
plt.semilogx(w_cheby1, 20 * np.log10(abs(h_cheby1)), label='Chebyshev I')
plt.semilogx(w_cheby2, 20 * np.log10(abs(h_cheby2)), label='Chebyshev II')
plt.semilogx(w_ellip, 20 * np.log10(abs(h_ellip)), label='Elliptical')
plt.title('Frequency response of filters')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', linestyle='-', color='grey')
plt.legend()

plt.show()

