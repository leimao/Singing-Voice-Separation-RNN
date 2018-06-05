

from download import download_mir1k
from preprocess import get_random_wav_batch, load_wavs, wavs_to_specs, sample_data_batch, sperate_magnitude_phase
from model import SVSRNN


import os
import librosa
import numpy as np

def train(random_seed = 0):

    np.random.seed(random_seed)

    # Download MIR1K dataset
    download_dir = 'download'
    data_dir = 'data'
    mir1k_dir = 'data/MIR1K'

    train_path = os.path.join(mir1k_dir, 'train.txt')
    valid_path = os.path.join(mir1k_dir, 'valid.txt')
    #mir1k_dir = download_mir1k(download_dir = download_dir, data_dir = data_dir)
    #wavs_dir = os.path.join(mir1k_dir, 'MIR-1K/UndividedWavfile')
    #wavs_dir = os.path.join(mir1k_dir, 'MIR-1K/Wavfile')

    with open(train_path, 'r') as text_file:
        content = text_file.readlines()
    wav_filenames_train = [file.strip() for file in content] 

    with open(valid_path, 'r') as text_file:
        content = text_file.readlines()
    wav_filenames_valid = [file.strip() for file in content] 

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    # Model parameters
    learning_rate = 0.0001
    num_rnn_layer = 3
    num_hidden_units = [256, 256, 256]
    batch_size = 64
    sample_frames = 10
    iterations = 50000
    tensorboard_directory = './graphs/svsrnn'
    log_directory = './log'
    train_log_filename = 'train_log.csv'
    clear_tensorboard = False
    model_directory = './model'
    model_filename = 'svsrnn.ckpt'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()

    # Load train wavs
    wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train, sr = mir1k_sr)

    # Turn waves to spectrums
    stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
        wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)


    wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames = wav_filenames_valid, sr = mir1k_sr)
    stfts_mono_valid, stfts_src1_valid, stfts_src2_valid = wavs_to_specs(
        wavs_mono = wavs_mono_valid, wavs_src1 = wavs_src1_valid, wavs_src2 = wavs_src2_valid, n_fft = n_fft, hop_length = hop_length)


    # Initialize model
    model =  SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)

    # Start training
    for i in (range(iterations)):
        
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono = stfts_mono_train, stfts_src1 = stfts_src1_train, stfts_src2 = stfts_src2_train, batch_size = batch_size, sample_frames = sample_frames)
        x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
        y1, _ = sperate_magnitude_phase(data = data_src1_batch)
        y2, _ = sperate_magnitude_phase(data = data_src2_batch)

        train_loss = model.train(x = x_mixed, y1 = y1, y2 = y2, learning_rate = learning_rate)

        if i % 10 == 0:
            print('Step: %d Train Loss: %f' %(i, train_loss))

        if i % 200 == 0:
            print('==============================================')
            data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
                stfts_mono = stfts_mono_valid, stfts_src1 = stfts_src1_valid, stfts_src2 = stfts_src2_valid, batch_size = batch_size, sample_frames = sample_frames)
            x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
            y1, _ = sperate_magnitude_phase(data = data_src1_batch)
            y2, _ = sperate_magnitude_phase(data = data_src2_batch)

            y1_pred, y2_pred, validation_loss = model.validate(x = x_mixed, y1 = y1, y2 = y2)
            print('Step: %d Validation Loss: %f' %(i, validation_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(i, train_loss, validation_loss))

        if i % 1000 == 0:
            model.save(directory = model_directory, filename = model_filename)



if __name__ == '__main__':
    
    train()
