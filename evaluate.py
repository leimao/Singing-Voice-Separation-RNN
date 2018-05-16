

import librosa
import os
import numpy as np
from preprocess import load_wavs, prepare_data_full, wavs_to_specs, sperate_magnitude_phase, combine_magnitdue_phase
from model import SVSRNN


def generate_demo():

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = 1024 // 4
    tensorboard_directory = 'graphs/svsrnn'
    clear_tensorboard = False
    model_directory = 'model'
    model_filename = 'svsrnn.ckpt'
    model_filepath = os.path.join(model_directory, model_filename)

    wav_file_test = ['small_test_data/yifen_4_10.wav']

    wavs_mono_test, wavs_src1_test, wavs_src2_test = load_wavs(filenames = wav_file_test, sr = mir1k_sr)

    print('y1 shape: ', wavs_src1_test[0].shape)
    print(wavs_src1_test[0][0:20])
    print('====================')
    np.where(wavs_src1_test[0] < 0)[0]
    print('====================')

    #wav_test, _ = librosa.load(wav_file_test, sr = sr, mono = False)
    #wav_src1_test = wav_test[0]
    #wav_src2_test = wav_test[1]
    #wav_mono_test = librosa.to_mono(wav_test) * 2

    stfts_mono_test, stfts_src1_test, stfts_src2_test = wavs_to_specs(
        wavs_mono = wavs_mono_test, wavs_src1 = wavs_src1_test, wavs_src2 = wavs_src2_test, n_fft = n_fft, hop_length = hop_length)

    #stft_mono_test = librosa.stft(wav_mono_test, n_fft = n_fft, hop_length = hop_length)
    #stft_src1_test = librosa.stft(wav_src1_test, n_fft = n_fft, hop_length = hop_length)
    #stft_src2_test = librosa.stft(wav_src2_test, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono_test, stfts_src1 = stfts_src1_test, stfts_src2 = stfts_src2_test)





    #magnitude_mono_test, phase_mono_test = sperate_magnitude_phase(data = stfts_mono_test)
    #magnitude_src1_test, phase_src1_test = sperate_magnitude_phase(data = stfts_src1_test)
    #magnitude_src2_test, phase_src2_test = sperate_magnitude_phase(data = stfts_src2_test)

    #stft_mono_magnitude_test = np.abs(stft_mono_test)
    #stft_mono_phase_test = np.angle(stft_mono_test)
    #stft_src1_magnitude_test = np.abs(stft_src1_test)
    #stft_src1_phase_test = np.angle(stft_src1_test)
    #stft_src2_magnitude_test = np.abs(stft_src2_test)
    #stft_src2_phase_test = np.angle(stft_src2_test)


    model = SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = 3, num_hidden_units = [256, 256, 256], tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)
    model.load(filepath = model_filepath)

    for stft_mono_full in stfts_mono_full:
        magnitude_mono, phase_mono = sperate_magnitude_phase(data = stft_mono_full)
        magnitude_mono = np.array([magnitude_mono])

        y1_pred, y2_pred = model.test(x = magnitude_mono)

        print('BBBB')
        print(y1_pred[0:20])
        print('BBBB')

        #print(y1_pred.shape)
        #print(y2_pred.shape)

        # ISTFT with the phase from mono
        #print(phase_mono)
        print('phase shape: ', phase_mono.shape)
        print('y1_pred shape: ', y1_pred.shape)

        y1_stft_hat = combine_magnitdue_phase(magnitudes = y1_pred[0], phases = phase_mono)
        y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = phase_mono)

        y1_stft_hat = y1_stft_hat.transpose()
        print('AAAAA')
        print(y1_stft_hat[0:20])
        print('AAAAA')
        y2_stft_hat = y2_stft_hat.transpose()

        y1_hat = librosa.istft(y1_stft_hat, hop_length = hop_length)
        y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)


        print('y1_hat shape: ', y1_hat.shape)
        print(y1_hat[0:20])

        # Don't do this, it makes the sound worse
        #y1_hat[y1_hat < 0] = 0
        #y2_hat[y2_hat < 0] = 0



        librosa.output.write_wav('test_separation_src1.wav', y1_hat, mir1k_sr)
        librosa.output.write_wav('test_separation_src2.wav', y2_hat, mir1k_sr)



if __name__ == '__main__':
    
    generate_demo()