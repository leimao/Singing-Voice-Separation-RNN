Singing_Voice_Separation_RNN


## Dependencies



* RarFile 3.0
* LibROSA 0.6
* ProgressBar2 3.37.1
* FFmpeg 4.0


## Caveates

### LibROSA


### AudioRead Problem on Ubuntu 16.04

```
Traceback (most recent call last):
  File "tutorial_1.py", line 12, in <module>
    y, sr = librosa.load(filename)
  File "/home/marine/anaconda3/lib/python3.6/site-packages/librosa/core/audio.py", line 112, in load
    with audioread.audio_open(os.path.realpath(path)) as input_file:
  File "/home/marine/anaconda3/lib/python3.6/site-packages/audioread/__init__.py", line 116, in audio_open
    raise NoBackendError()
audioread.NoBackendError
```

Install 

```
# Anaconda
$ conda install -c conda-forge ffmpeg
# apt-get
apt-get install ffmpeg
```

https://github.com/librosa/librosa#audioread

## References

https://github.com/andabi/music-source-separation