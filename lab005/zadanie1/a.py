import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define poles and zeros
poles = np.array([-0.5 + 9.5j, -0.5 - 9.5j, -1 + 10j, -1 - 10j, -0.5 + 10.5j, -0.5 - 10.5j])
zeros = np.array([0 + 5j, 0 - 5j, 0 + 15j, 0 - 15j])

# Function to plot poles and zeros

def plot_poles_zeros(poles, zeros):
    plt.figure(figsize=(8, 8))
    plt.scatter(poles.real, poles.imag, marker='x', color='red', label='Poles')
    plt.scatter(zeros.real, zeros.imag, marker='o', color='blue', label='Zeros')
    plt.title('Poles and Zeros in the Complex Plane')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid()
    plt.legend()
    plt.show()


# Function to plot amplitude and phase response
def plot_response(omega, H_s):
    amplitude = np.abs(H_s)
    phase = np.angle(H_s)
    amplitude_dB = 20 * np.log10(amplitude)

    plt.figure(figsize=(12, 6))
    plt.plot(omega, amplitude, label='|H(jω)|')
    plt.title('Amplitude-Frequency Response')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(omega, amplitude_dB, label='20log|H(jω)| (dB)')
    plt.title('Amplitude-Frequency Response')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(omega, phase, label='Phase of H(jω)')
    plt.title('Phase-Frequency Response')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (radians)')
    plt.grid()
    plt.legend()
    plt.show()


# Calculate transfer function
def calculate_transfer_function(poles, zeros, omega):
    numerator_coeffs = np.poly(zeros)   # współczynniki wielomianu ( wyzerowanie i wzmocnienie )
    denominator_coeffs = np.poly(poles)
    s = 1j * omega
    H_s = np.polyval(numerator_coeffs, s) / np.polyval(denominator_coeffs, s)
    return H_s


# Define frequency range for plotting
omega = np.linspace(0, 20, 1000)


# Plot poles and zeros
plot_poles_zeros(poles, zeros)

# Calculate transfer function
H_s = calculate_transfer_function(poles, zeros, omega)

# Plot amplitude and phase response
plot_response(omega, H_s)

# Calculate band characteristics
stop_band_attenuation = np.max(20 * np.log10(np.abs(calculate_transfer_function(poles, zeros, np.linspace(0, 4, 500)))))
pass_band_gain = np.max(np.abs(calculate_transfer_function(poles, zeros, np.linspace(9, 11, 200))))

print(f'Max attenuation in the stop band: {stop_band_attenuation} dB')
print(f'Gain in the pass band: {pass_band_gain}')

plot_response(omega, 1/pass_band_gain * H_s)
# plt.plot(omega, np.abs(1/pass_band_gain * H_s), label='|H(jω)|')
# plt.grid()
# plt.legend()
# plt.show()
#
# Z wykresu amplituda-częstotliwość możemy stwierdzić, że system jest rzeczywiście filtrem pasmowo-przepustowym,
# ponieważ umożliwia przejście częstotliwości około 10 rad/s przy mniejszym tłumieniu w porównaniu z innymi częstotliwościami.
