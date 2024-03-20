import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def moja_korelacja(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.correlate(a_flat, b_flat, mode='full')


# Load the MAT file
mat_contents = scipy.io.loadmat('adsl_x.mat')

K = 4
M = 32
N = 512

signal = mat_contents['x']
correlation_result = moja_korelacja(signal[:32], signal)
new_prefix_list = []
last_index = -1000

for index, value in enumerate(correlation_result):
    if value > 600:
        if index - last_index < M:
            continue
        new_prefix_list.append(index)
        last_index = index

print(new_prefix_list, "\n", len(new_prefix_list))
# Plot the result to visualize
plt.figure(figsize=(10, 5))
plt.plot(correlation_result)
plt.title('Cross-Correlation between Prefix and Signal')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()