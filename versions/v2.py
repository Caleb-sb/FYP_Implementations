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

    # def process(self):
    #     # Save the buffer state in case it changes while processing
    #     # interference[1] = np.array([i for i in self.to_process[1]])
    #     # Process the buffers
    #     # u_scale = np.max(self.to_process[0])
    #     d_scale = np.max(self.to_process[1])
    #
    #     for i in self.to_process[0]:
    #         self.u[self.p_index0] = i
    #         self.p_index0 += 1
    #     for i in self.to_process[1]:
    #         self.d[self.p_index1] = i
    #         self.p_index1 += 1
    #     # self.d.append([item for item in self.to_process[1]])
    #
    #     # if (self.p_index==0):
    #     #     out, error, self.h = NLMS_filter(
    #     #                     taps = self.order,
    #     #                     u = u/u_scale,
    #     #                     d = d/d_scale,
    #     #                     music=np.zeros(50),
    #     #                     h=self.h)
    #     # #
    #     # else:
    #     #     out, error, self.h = NLMS_filter(
    #     #                     taps = self.order,
    #     #                     u = u[self.p_index-self.order+1:u.size]/u_scale,
    #     #                     d = d[self.p_index-self.order+1:d.size]/d_scale,
    #     #                     music=np.zeros(50),
    #     #                     h=self.h)
    #     # print("THIS IS OUTPUT SIZE: ", out.size)
    #     print("This is index: ", self.p_index0)
    #     # if (self.p_index == 0):
    #     #     for i in out:
    #     #         self.y[0].append(i*d_scale)
    #     #     for e in error:
    #     #         self.e[0].append(e)
    #     # else:
    #     #     self.y.append([y*d_scale for y in out])
    #     #     self.e.append([e for e in error])
    #     # self.p_index = self.p_index + self.to_process[1].size
    #     # u = np.array(self.y, dtype=np.int16).reshape((np.array(self.y).size))
    #     if (self.p_index0 > self.duration*44100):
    #         return 0
