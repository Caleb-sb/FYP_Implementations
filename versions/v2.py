import pyaudio
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from util.adaptive import NLMS_filter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf


class Version2:
    def __init__(self, micsPort, duration, ff_filename, fb_filename):
        # Places to save the instance's recordings
        self.ff_filename = ff_filename
        self.fb_filename = fb_filename

        # Sampling rate
        self.rate = 44100
        self.micsPort = micsPort
        # PyAudio Objects for recording audio
        self.audio_io = pyaudio.PyAudio()
        self.channels = 2

        # The stereo signals of the mics are stored here
        self.mics  = np.array([np.zeros(duration*self.rate, dtype=np.int16),
                              np.zeros(duration*self.rate, dtype=np.int16)])
        # Index for writing to mic array
        self.index = 0

    def callback(self, in_data, frame_count, time_info, status):
        noise = np.frombuffer(in_data, dtype=np.int16)
        # Splitting the channels
        self.mics[0][self.index:self.index+int(noise.size/2)] = noise[::2]
        self.mics[1][self.index:self.index+int(noise.size/2)] = noise[1::2]
        # Updating the index
        self.index = self.index + int(noise.size/2)
        # Completing the function signature with the return
        return (in_data, pyaudio.paContinue)

    def start(self):
        # Initing the stream
        self.micStream = self.audio_io.open(
            format   = pyaudio.paInt16,
            channels = self.channels,
            rate     = self.rate,
            input_device_index=self.micsPort, # sennheiser
            input = True,
            stream_callback=self.callback
        )
        # Starting the stream
        self.micStream.start_stream()

        # Wait for interrupt or end of array
        print('\nI\'m Listening...')
        while self.micStream.is_active():
            try:
                time.sleep(1)
            except ValueError:
                print("Arrays filled. Saving to wav file...")
                break
            except KeyboardInterrupt:
                print("Interrupted by user. Saving to wav file...")
                break
        self.stop()

    def stop(self):
        # PyAudio cleanup
        self.micStream.close()
        self.audio_io.terminate()
        # Saving the recorded signals
        wf.write(self.fb_filename, 44100, self.mics[0])
        wf.write(self.ff_filename, 44100, self.mics[1])
