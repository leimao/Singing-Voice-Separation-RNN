

import librosa
import os
import numpy as np
from preprocess import load_wavs, prepare_data_full, wavs_to_specs, sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN


def generate_demo():

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = [256, 256, 256]
    tensorboard_directory = 'graphs/svsrnn'
    clear_tensorboard = False
    model_directory = 'model'
    model_filename = 'svsrnn.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)

    wav_filenames = ['small_test_data/yifen_4_10.wav', 'small_test_data/yifen_5_10.wav']
    output_directory = 'demo'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
        wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)

    model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load(filepath = model_filepath)

    for wav_filename, wav_mono, stft_mono_full in zip(wav_filenames, wavs_mono, stfts_mono_full):

        stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono_full)
        stft_mono_magnitude = np.array([stft_mono_magnitude])

        y1_pred, y2_pred = model.test(x = stft_mono_magnitude)

        # ISTFT with the phase from mono
        y1_stft_hat = combine_magnitdue_phase(magnitudes = y1_pred[0], phases = stft_mono_phase)
        y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase)

        y1_stft_hat = y1_stft_hat.transpose()
        y2_stft_hat = y2_stft_hat.transpose()

        y1_hat = librosa.istft(y1_stft_hat, hop_length = hop_length)
        y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)

        wav_filename_dir = os.path.dirname(wav_filename)
        wav_filename_base = os.path.basename(wav_filename)
        wav_mono_filename = wav_filename_base.split('.')[0] + '_mono' + wav_filename_base.split('.')[1]
        wav_src1_hat_filename = wav_filename_base.split('.')[0] + '_src1' + wav_filename_base.split('.')[1]
        wav_src2_hat_filename = wav_filename_base.split('.')[0] + '_src2' + wav_filename_base.split('.')[1]
        wav_mono_filepath = os.path.join(output_directory, wav_mono_filename)
        wav_src1_hat_filepath = os.path.join(output_directory, wav_src1_hat_filename)
        wav_src2_hat_filepath = os.path.join(output_directory, wav_src2_hat_filename)

        librosa.output.write_wav(wav_mono_filepath, wav_mono, mir1k_sr)
        librosa.output.write_wav(wav_src1_hat_filepath, y1_hat, mir1k_sr)
        librosa.output.write_wav(wav_src2_hat_filepath, y2_hat, mir1k_sr)



if __name__ == '__main__':
    
    generate_demo()