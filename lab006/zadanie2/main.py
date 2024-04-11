import librosa
import numpy as np
import codes
import filters
import utilities

audio_path = r'./data/'


def get_non_silent_chunks(audio: np.ndarray) -> list:
    """
    function splits audio on silent parts to make it possible to analize code
    :param audio: he input audio signal.
    :return: list of audio chunks
    """
    noise_lvl = utilities.estimate_noise_level(audio)
    non_silent_intervals = librosa.effects.split(audio, top_db=-noise_lvl, hop_length=64, frame_length=128)
    audio_segments = []
    for start_idx, end_idx in non_silent_intervals:
        segment = audio[start_idx:end_idx]
        audio_segments.append(segment)
    return audio_segments


def main(file: str) -> None:
    """
    main function, prints DTMF code from audio file
    :param file: name of audio file located in ./data/
    :return:
    """
    audio, fs = librosa.load(audio_path + file)
    audio = filters.select_freq(audio, fs)

    #
    # for i in range(3):
    #     audio = utilities.spectral_subtraction_noise(audio, fs)

    audio_chunks = get_non_silent_chunks(audio)
    code = ''
    for chunk in audio_chunks:
        code += codes.extract_number(chunk, fs)
    print(code)

    # # audio = utilities.segment_normalization(audio, fs)


if __name__ == "__main__":
    # main(r'challenge 2022.wav')
    for i in range(10):
        main(rf's{i}.wav')
