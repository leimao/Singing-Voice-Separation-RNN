
import numpy as np
import librosa


def load_wavs(filenames, sr):

    wavs_mono = list()
    wavs_src1 = list()
    wavs_src2 = list()

    for filename in filenames:
        wav, _ = librosa.load(filename, sr = sr, mono = False)
        assert (wav.ndim == 2) and (wav.shape[0] == 2), 'Require wav to have two channels'
        wav_mono = librosa.to_mono(wav) * 2 # Cancelling average
        wav_src1 = wav[0, :]
        wav_src2 = wav[1, :]
        wavs_mono.append(wav_mono)
        wavs_src1.append(wav_src1)
        wavs_src2.append(wav_src2)

    return wavs_mono, wavs_src1, wavs_src2


def load_mono_wavs(filenames, sr):

    wavs_mono = list()
    for filename in filenames:
        wav_mono, _ = librosa.load(filename, sr = sr, mono = True)
        wavs_mono.append(wav_mono)

    return wavs_mono



def wavs_to_specs(wavs_mono, wavs_src1, wavs_src2, n_fft = 1024, hop_length = None):

    stfts_mono = list()
    stfts_src1 = list()
    stfts_src2 = list()

    for wav_mono, wav_src1, wav_src2 in zip(wavs_mono, wavs_src1, wavs_src2):
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stft_src1 = librosa.stft(wav_src1, n_fft = n_fft, hop_length = hop_length)
        stft_src2 = librosa.stft(wav_src2, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono)
        stfts_src1.append(stft_src1)
        stfts_src2.append(stft_src2)

    return stfts_mono, stfts_src1, stfts_src2

def prepare_data_full(stfts_mono, stfts_src1, stfts_src2):

    stfts_mono_full = list()
    stfts_src1_full = list()
    stfts_src2_full = list()

    for stft_mono, stft_src1, stft_src2 in zip(stfts_mono, stfts_src1, stfts_src2):
        stfts_mono_full.append(stft_mono.transpose())
        stfts_src1_full.append(stft_src1.transpose())
        stfts_src2_full.append(stft_src2.transpose())

    return stfts_mono_full, stfts_src1_full, stfts_src2_full


def sample_data_batch(stfts_mono, stfts_src1, stfts_src2, batch_size = 64, sample_frames = 8):

    stft_mono_batch = list()
    stft_src1_batch = list()
    stft_src2_batch = list()

    collection_size = len(stfts_mono)
    collection_idx = np.random.choice(collection_size, batch_size, replace = True)
    for idx in collection_idx:
        stft_mono = stfts_mono[idx]
        stft_src1 = stfts_src1[idx]
        stft_src2 = stfts_src2[idx]
        num_frames = stft_mono.shape[1]
        assert  num_frames >= sample_frames
        start = np.random.randint(num_frames - sample_frames + 1)
        end = start + sample_frames
        stft_mono_batch.append(stft_mono[:,start:end])
        stft_src1_batch.append(stft_src1[:,start:end])
        stft_src2_batch.append(stft_src2[:,start:end])

    # Shape: [batch_size, n_frequencies, n_frames]
    stft_mono_batch = np.array(stft_mono_batch)
    stft_src1_batch = np.array(stft_src1_batch)
    stft_src2_batch = np.array(stft_src2_batch)
    # Shape for RNN: [batch_size, n_frames, n_frequencies]
    data_mono_batch = stft_mono_batch.transpose((0, 2, 1))
    data_src1_batch = stft_src1_batch.transpose((0, 2, 1))
    data_src2_batch = stft_src2_batch.transpose((0, 2, 1))

    return data_mono_batch, data_src1_batch, data_src2_batch

def sperate_magnitude_phase(data):

    return np.abs(data), np.angle(data)

def combine_magnitdue_phase(magnitudes, phases):

    return magnitudes * np.exp(1.j * phases)


def specs_to_wavs_istft_batch(magnitudes, phases, hop_length):

    stft_matrices = combine_magnitdue_phase(magnitudes = magnitudes, phases = phases)

    wavs = list()
    for magnitude, phase in zip(magnitudes, phases):
        wav = librosa.istft(stft_matrices, hop_length = hop_length)
        wavs.append(wav)

    wavs = np.array(wavs)

    return wavs


def specs_to_wavs_griffin_lim_batch():

    # Recover an audio signal given only the magnitude of its Short-Time Fourier Transform (STFT)

    return

































def get_random_wav(filename, sr, duration):

    # Get a random range from wav

    wav, _ = librosa.load(filename, sr = sr, mono = False)
    print(wav)
    assert (wav.ndim == 2) and (wav.shape[0] == 2), 'Require wav to have two channels'

    wav_pad = pad_wav(wav = wav, sr = sr, duration = duration)
    wav_sample = sample_range(wav = wav, sr = sr, duration = duration)

    wav_sample_mono = librosa.to_mono(wav_sample)
    wav_sample_src1 = wav_sample[0, :]
    wav_sample_src2 = wav_sample[1, :]

    return wav_sample_mono, wav_sample_src1, wav_sample_src2


def get_random_wav_batch(filenames, sr, duration):

    # Get a random wav dataset of certain length

    wav_mono = list()
    wav_src1 = list()
    wav_src2 = list()

    for filename in filenames:
        wav_sample_mono, wav_sample_src1, wav_sample_src2 = get_random_wav(filename = filename, sr = sr, duration = duration)
        wav_mono.append(wav_sample_mono)
        wav_src1.append(wav_sample_src1)
        wav_src2.append(wav_sample_src2)

    wav_mono = np.array(wav_mono)
    wav_src1 = np.array(wav_src1)
    wav_src2 = np.array(wav_src2)

    return wav_mono, wav_src1, wav_src2

def wav_to_spec_batch(wavs, n_fft, hop_length = None):

    # Short-time Fourier transform (STFT) for wav matrix in batch
    # n_fft : int > 0 [scalar] FFT window size.
    # hop_length : int > 0 [scalar] number audio of frames between STFT columns. If unspecified, defaults win_length / 4.

    assert (wavs.ndim == 2), 'Single wav uses librosa.stft() directly'

    stft_matrices = list()

    for wav in wavs:
        stft_matrix = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stft_matrices.append(stft_matrix)

    stft_matrices = np.array(stft_matrices)

    return stft_matrices


def spec_to_wav_batch(stft_matrices, hop_length = None):

    # Every stft matrix in stft matrices may have complex numbers

    assert (stft_matrices.ndim == 3), 'Single stft maxtrix uses librosa.istft() directly'

    wavs = list()

    for stft_matrix in stft_matrices:
        wav = librosa.istft(stft_matrix, hop_length = hop_length)
        wavs.append(wav)

    wavs = np.array(wavs)

    return wavs

def get_spec_freq(stft_matrix, sr, n_fft):

    # Get the sample frequencies for stft_matrix

    assert (stft_matrix.ndim == 2)

    return np.arange(stft_matrix.shape[0]) / n_fft * sr


def get_magnitude(x):

    # Get magnitude of complex scalar, vector or matrix

    return np.abs(x)

def get_phase(x):

    # Get phase of complex scalar, vector or matrix

    return np.angle(x)

def make_complex(magnitude, phase):

    # Make complex using magnitude and phase

    return magnitude * np.exp(1.j * phase)


def pad_wav(wav, sr, duration):

    # Pad short wav with zeros at the end so that wav is long enough for model training

    # Only pad mono sourced or dual sourced wav
    assert(wav.ndim <= 2)

    # Minimum length of wav
    n_samples = sr * duration

    # Number of elements to pad per source
    pad_len = np.maximum(0, n_samples - wav.shape[-1])

    if wav.ndim == 1:
        pad_width = (0, pad_len)
    else:
        pad_width = ((0, 0), (0, pad_len))

    wav  = np.pad(wav, pad_width = pad_width, mode = 'constant', constant_values = 0)

    return wav


def sample_range(wav, sr, duration):

    # Down sample wav to certain length

    assert(wav.ndim <= 2)

    # Target length must be shorter than wav length
    wav_len = wav.shape[-1]
    target_len = sr * duration
    assert(target_len <= wav_len), 'wav too short to sample'

    # Randomly choose sampling range
    start = np.random.randint(wav_len - target_len + 1)
    end = start + target_len

    if wav.ndim == 1:
        wav_sample = wav[start:end]
    else:
        wav_sample = wav[:, start:end]

    return wav_sample

