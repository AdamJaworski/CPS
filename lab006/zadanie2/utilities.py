import matplotlib.pyplot as plt
import numpy as np
import librosa


def display_freq(audio: np.ndarray, fs) -> None:
    fft_result = np.fft.fft(audio)
    fft_freq = np.fft.fftfreq(len(audio), 1 / fs)

    # Taking the magnitude of the FFT result (for volume) and only the first half (due to symmetry)
    n = len(fft_result) // 2
    fft_magnitude = np.abs(fft_result[:n]) * 2 / len(audio)

    x = 650  # Lower frequency limit
    y = 1500  # Upper frequency limit
    indices = np.where((fft_freq >= x) & (fft_freq <= y))[0]
    # Plotting the Frequency vs Volume (Amplitude) graph
    plt.figure(figsize=(14, 6))
    plt.plot(fft_freq[indices], fft_magnitude[indices])
    plt.xticks(np.arange(x, y, 50))
    plt.title('Frequency vs Volume (Amplitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Volume (Amplitude)')
    plt.grid(True)
    plt.show()


def normalize_audio(audio: np.ndarray, target_peak: float = 0.1) -> np.ndarray:
    """
    Normalize the audio signal so that its maximum peak is at the target peak level.
    :param audio: The input audio signal.
    :param target_peak: The target peak level.
    :return: The normalized audio signal.
    """

    max_peak = np.abs(audio).max()
    if max_peak == 0:
        return audio
    normalization_factor = target_peak / max_peak
    return audio * normalization_factor


def segment_normalization(audio, fs, segment_length=1.0, target_peak=0.15):
    """
    Normalize audio in segments to achieve a more uniform volume level across its duration.

    Parameters:
    audio (numpy array): Input audio signal.
    fs (int): Sampling rate of the audio signal.
    segment_length (float): Length of each segment to normalize, in seconds.
    target_peak (float): Target peak level for normalization.

    Returns:
    numpy array: The audio signal with normalized volume across segments.
    """
    samples_per_segment = int(segment_length * fs)
    num_segments = int(np.ceil(len(audio) / samples_per_segment))
    normalized_audio = np.zeros_like(audio)

    for i in range(num_segments):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        segment = audio[start_idx:end_idx]
        max_peak = np.abs(segment).max()
        if max_peak == 0:
            continue  # Avoid division by zero for silent segments
        normalization_factor = target_peak / max_peak
        normalized_audio[start_idx:end_idx] = segment * normalization_factor

    return normalized_audio


def estimate_noise_level(audio, frame_length=2048, hop_length=512):
    energy = np.array([
        sum(abs(audio[i:i + frame_length]) ** 2)
        for i in range(0, len(audio), hop_length)
    ])

    energy_db = librosa.power_to_db(energy, ref=np.max)

    # Assume the lower 10th percentile of energy frames are noise
    noise_frames_db = np.percentile(energy_db, 10)

    return noise_frames_db


def spectral_subtraction_noise(signal, sr, n_fft=2048, hop_length=512):
    """
    Perform spectral subtraction assuming the last second of the audio is noise.

    Parameters:
    - signal: The input audio signal (numpy array).
    - sr: Sampling rate of the audio signal.
    - n_fft: The number of data points used in each block for the FFT.
    - hop_length: The number of samples between successive frames.

    Returns:
    - The denoised audio signal (numpy array).
    """
    # Estimate noise from the last second of the audio
    noise_frames = sr // hop_length
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    noise_estimation = np.mean(np.abs(stft[:, -noise_frames:]), axis=1, keepdims=True)

    # Perform spectral subtraction
    subtracted_magnitude = np.maximum(np.abs(stft) - noise_estimation, 0)
    phase = np.angle(stft)
    denoised_stft = subtracted_magnitude * np.exp(1j * phase)

    # Inverse STFT to convert back to time domain
    denoised_signal = librosa.istft(denoised_stft, hop_length=hop_length)

    return denoised_signal