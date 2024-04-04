import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lti, impulse, step, freqs

# Given values
omega_c = 2 * np.pi * 100  # cutoff frequency in rad/s


# Define the function to calculate the poles of the Butterworth filter
def butterworth_poles(n, omega_c):
    return [omega_c * np.exp(1j * ((np.pi / 2) + (np.pi / (2 * n)) + (((k - 1) * np.pi) / n))) for k in range(n)]


# Calculate poles for filter orders 2, 4, 6, 8
poles = {n: butterworth_poles(n, omega_c) for n in [2, 4, 6, 8]}

# Plotting the poles in the complex plane
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, (n, p) in zip(axs.flatten(), poles.items()):
    # Plot the unit circle
    circle = plt.Circle((0, 0), omega_c, color='blue', fill=False)
    ax.add_artist(circle)

    # Plot the poles
    ax.plot(np.real(p), np.imag(p), 'x', markersize=10)
    ax.set_title(f'Poles for n={n}')
    ax.grid(True)
    ax.set_xlim([-omega_c - 100, omega_c + 100])
    ax.set_ylim([-omega_c - 100, omega_c + 100])
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

# Show the pole plots
plt.tight_layout()
plt.show()


# Function to calculate frequency response given the poles
def frequency_response(poles, omega):
    num = np.polyval([1], 1j * omega)
    den = np.polyval(np.poly(poles), 1j * omega)
    H = num / den
    return H


# # Define frequency range for plotting: from 0.1 to 10 times the cutoff frequency
omega = np.logspace(np.log10(omega_c / 10), np.log10(omega_c * 10), 1000)
f = omega / (2 * np.pi)  # Convert omega to Hz for the frequency axis

# Plot amplitude and phase response on one plot for all filter orders
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

colors = ['b', 'g', 'r', 'c']  # Colors for each filter order
for (n, p), color in zip(poles.items(), colors):
    H = frequency_response(p, omega)

    # Amplitude response (in dB)
    axs[0].semilogx(f, 20 * np.log10(abs(H)), label=f'n={n}', color=color)
    axs[0].set_title('Amplitude Responses for n=2, 4, 6, 8 (20log10(|H(jw)|)')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel('Amplitude [dB]')
    axs[0].grid(which='both', axis='both')
    axs[0].axvline(100, color='black', linestyle='--')
    axs[0].legend()

    # Phase response
    axs[1].semilogx(f, np.angle(H, deg=True), label=f'n={n}', color=color)
    axs[1].set_title('Phase Responses for n=2, 4, 6, 8')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Phase [degrees]')
    axs[1].grid(which='both', axis='both')
    axs[1].axvline(100, color='black', linestyle='--')
    axs[1].legend()

# Show the plots
plt.tight_layout()
plt.show()

# Order and cutoff frequency for the filter
N = 4

# Get the filter coefficients in terms of zeros, poles, and gain
z, p, k = butter(N, omega_c, analog=True, output='zpk')

# Convert zeros, poles, and gain to transfer function (B, A) representation
B, A = butter(N, omega_c, analog=True, output='ba')

# Create a Linear Time Invariant system
system = lti(B, A)

# Calculate impulse response
t_impulse, imp = impulse(system)

# Calculate step response
t_step, step_resp = step(system)

# Plot the impulse and step responses
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Impulse response
axs[0].plot(t_impulse, imp)
axs[0].set_title('Impulse Response of 4th Order Butterworth Filter')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Amplitude')
axs[0].grid()

# Step response
axs[1].plot(t_step, step_resp)
axs[1].set_title('Step Response of 4th Order Butterworth Filter')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Amplitude')
axs[1].grid()

# Show the plots
plt.tight_layout()
plt.show()
