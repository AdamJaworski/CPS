import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lti, impulse, step, freqs

# Given values
omega_c = 2 * np.pi * 100  # cutoff frequency in rad/s


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

N = [2, 4, 6, 8]
for n in N:
    b, a = butter(n, omega_c, 'low', analog=True)
    w, h = freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))          # PLOT zamiana
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency

plt.show()

N = [2, 4, 6, 8]
for n in N:
    b, a = butter(n, omega_c, 'low', analog=True)
    w, h = freqs(b, a)
    plt.semilogx(w, np.angle(h))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency

plt.show()


# Convert zeros, poles, and gain to transfer function (B, A) representation
B, A = butter(4, omega_c, analog=True, output='ba')

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