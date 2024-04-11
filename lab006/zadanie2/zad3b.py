import librosa
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

data_path = r'./data/'

keys = {
    (697, 1209): '1',
    (697, 1336): '2',
    (697, 1477): '3',
    (770, 1209): '4',
    (770, 1336): '5',
    (770, 1477): '6',
    (852, 1209): '7',
    (852, 1336): '8',
    (852, 1477): '9',
    (941, 1209): '*',
    (941, 1336): '0',
    (941, 1477): '#'
}


def get_freq(low_f, high_f) -> str:
    low_v  = np.array([697, 770, 852, 941])
    high_v = np.array([1209, 1336, 1477])

    low_min = abs(low_v - low_f)
    high_min = abs(high_v - high_f)

    if np.min(low_min) > 3:
        return ''
    if np.min(high_min) > 3:
        return ''

    low_f = low_v[np.where(low_min == np.min(low_min))][0]
    high_f = high_v[np.where(high_min == np.min(high_min))][0]

    return keys[(low_f, high_f)]


for file in range(0,1):
# for file in range(0, 10):

    audio, fs = librosa.load(f'{data_path}challenge 2022.wav', sr=None)
    # audio, fs = librosa.load(f'{data_path}s{file}.wav')
    # non_silent_intervals = librosa.effects.split(audio, top_db=-15)

    audio_segments = []
    # for start_idx, end_idx in non_silent_intervals:
    #     segment = audio[start_idx:end_idx]
    #     audio_segments.append(segment)

    for i in range(1, int((len(audio) / (4 * fs)))):
        audio_segments.append(audio[(i-1) * (4 * fs) : i * (4 * fs)])

    for chunk in audio_segments:
        fft_result = np.fft.fft(chunk)
        fft_freq = np.fft.fftfreq(len(chunk), 1 / fs)

        n = len(fft_result) // 2
        fft_magnitude = np.abs(fft_result[:n]) * 2 / len(chunk)
        print(fft_magnitude.mean())

    code = ''
    for i, chunk in enumerate(audio_segments):
        fft_result = np.fft.fft(chunk)
        fft_freq = np.fft.fftfreq(len(chunk), 1 / fs)

        n = len(fft_result) // 2
        fft_magnitude = np.abs(fft_result[:n]) * 2 / len(chunk)
        fft_freq = fft_freq[:n]  # Only consider the first half of the frequencies

        low_indices = np.where((fft_freq >= 680) & (fft_freq < 1000))[0]
        high_indices = np.where((fft_freq >= 1200) & (fft_freq <= 1500))[0]

        if low_indices.size > 0:
            low_f_index = low_indices[np.argmax(fft_magnitude[low_indices])]
            low_f = fft_freq[low_f_index]
        else:
            low_f = None

        if high_indices.size > 0:
            high_f_index = high_indices[np.argmax(fft_magnitude[high_indices])]
            high_f = fft_freq[high_f_index]
        else:
            high_f = None

        if low_f and high_f:
            code += get_freq(low_f, high_f)
        else:
            print(':<')
    print(f"{file}. {code}")