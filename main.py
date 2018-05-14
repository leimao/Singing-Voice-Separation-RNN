

from download import download_mir1k
from preprocess import get_random_wav_batch, load_wavs, wavs_to_specs, sample_data_batch, sperate_magnitude_phase
from model import SVSRNN


import os
import librosa

def main():

    # Download MIR1K dataset
    download_dir = 'download/'
    data_dir = 'data/'
    mir1k_dir = download_mir1k(download_dir = download_dir, data_dir = data_dir)
    #wavs_dir = os.path.join(mir1k_dir, 'MIR-1K/UndividedWavfile')
    wavs_dir = os.path.join(mir1k_dir, 'MIR-1K/Wavfile')
    #print(wavs_dir)

    wav_filenames = list()
    for file in os.listdir(wavs_dir):
        if file.endswith('.wav'):
            wav_filenames.append(os.path.join(wavs_dir, file))

    #wav_filenames = wav_filenames[0:2]

    wav_filenames_train = wav_filenames[:2]
    wav_filenames_valid = wav_filenames[-2:]


    mir1k_sr = 16000
    n_fft = 1024
    hop_length = 1024 // 4
    learning_rate = 0.001
    batch_size = 64
    sample_frames = 4

    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames_train, sr = mir1k_sr)

    wavs_mono[0] = wavs_mono[0][0:102400]
    wavs_mono[1] = wavs_mono[1][0:102400]
    wavs_src1[0] = wavs_src1[0][0:102400]
    wavs_src1[1] = wavs_src1[1][0:102400]
    wavs_src2[0] = wavs_src2[0][0:102400]
    wavs_src2[1] = wavs_src2[1][0:102400]

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    print(stfts_mono[0].shape)




    wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames = wav_filenames_valid, sr = mir1k_sr)
    stfts_mono_valid, stfts_src1_valid, stfts_src2_valid = wavs_to_specs(wavs_mono = wavs_mono_valid, wavs_src1 = wavs_src1_valid, wavs_src2 = wavs_src2_valid, n_fft = n_fft, hop_length = hop_length)







    #data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2, batch_size = 64, sample_frames = 8)


    #print(data_mono_batch.shape, data_src1_batch.shape, data_src2_batch.shape)


    #magnitude_mono_batch, phase_mono_batch = sperate_magnitude_phase(data = data_mono_batch)
    #magnitude_src1_batch, phase_src1_batch = sperate_magnitude_phase(data = data_src1_batch)
    #magnitude_src2_batch, phase_src2_batch = sperate_magnitude_phase(data = data_src2_batch)

    model =  SVSRNN(num_features = n_fft // 2 + 1, num_rnn_layer = 3, num_hidden_units = [256, 256, 256])


    #data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
    #    stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2, batch_size = 64, sample_frames = 4)
    #x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
    #x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
    #y1, _ = sperate_magnitude_phase(data = data_src1_batch)
    #y2, _ = sperate_magnitude_phase(data = data_src2_batch)

    for i in (range(20000)):
        
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2, batch_size = batch_size, sample_frames = sample_frames)
        x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
        y1, _ = sperate_magnitude_phase(data = data_src1_batch)
        y2, _ = sperate_magnitude_phase(data = data_src2_batch)
        #print(y2)
        
        train_loss = model.train(x = x_mixed, y1 = y1, y2 = y2, learning_rate = learning_rate)
        #y1_pred, y2_pred, validate_loss = model.validate(x = x_mixed, y1 = y1, y2 = y2)
        #print(x_mixed[0,0,0:10])
        #print(y1_pred[0,0,0:10])
        #print(y2_pred[0,0,0:10])

        #print(train_loss)

        if i % 10 == 0:
            print('==============================================================')
            print('Step: %d' %i)

            print("Train Loss: %f" %train_loss)
            y1_pred, y2_pred, validate_loss = model.validate(x = x_mixed, y1 = y1, y2 = y2)
            print(2 * x_mixed[0,0,0:5])
            print(y1[0,0,0:5] + y2[0,0,0:5])
            #print(y1_pred[0,0,0:5])
            #print(y2[0,0,0:5])
            #print(y2_pred[0,0,0:5])


            #data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            #    stfts_mono = stfts_mono_valid, stfts_src1 = stfts_src1_valid, stfts_src2 = stfts_src2_valid, batch_size = batch_size, sample_frames = sample_frames)
            #x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
            #x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
            #y1, _ = sperate_magnitude_phase(data = data_src1_batch)
            #y2, _ = sperate_magnitude_phase(data = data_src2_batch)


            #y1_pred, y2_pred, validate_loss = model.validate(x = x_mixed, y1 = y1, y2 = y2)

            #print("Validation Loss: %f" %validate_loss)
            #print(y1_pred - y1)
            #print(y1_pred)






    #sample_duration = 20

    #y, sr = librosa.load(wav_filenames[0], sr = None, mono = False)
    #print(sr)
    # Instead of loading all the data to memory at one time, we sample part of them to save memory
    # Each of them has shape [batch_size, n_frames]
    #wav_mono, wav_src1, wav_src2 = get_random_wav_batch(filenames = wav_filenames, sr = mir1k_sr, duration = 20)
    #print(wav_mono.shape)
    #print(wav_src1.shape)
    #print(wav_src2.shape)


    # For RNN input, the input shape is [batch_size, n_frames, n_frequencies]

    #print(os.path.join(wavs_dir, file))
    #wav_filenames = glob.glob('wavs_dir/*.wav')
    #print(wav_filenames)


    #get_random_wav_batch(filenames, sr, duration)












if __name__ == '__main__':
    
    main()
