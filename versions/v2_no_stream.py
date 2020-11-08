# Collects interference audio from microphone for processing
import pyaudio
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import os
from sklearn.metrics import mean_squared_error
from matplotlib import font_manager as fm, rcParams
import matplotlib
import noisereduce as nr

class MicsInput:

    def __init__(self, duration, ff_micPort, fb_micPort):
        FORMAT      = pyaudio.paInt16   #   audio format (bytes per sample?)
        self.CHANNELS    = 1                 #   single channel for microphone
        RATE        = 44100             #   samples per second
        self.noise_len   = duration*RATE

        self.mics   = np.array([np.zeros(self.noise_len, dtype=np.int16), np.zeros(self.noise_len, dtype=np.int16)])
        self.index0 = 0
        self.index1 = 0

        self.ff = pyaudio.PyAudio()
        self.fb = pyaudio.PyAudio()

        # stream object to get data from microphone
        self.ff_mic = self.ff.open(
            format  =FORMAT,
            channels=self.CHANNELS,
            rate    =RATE,
            input_device_index=ff_micPort, # sennheiser
            input=True,
            stream_callback=self.callback
        )
        self.fb_mic = self.fb.open(
            format=FORMAT,
            channels=self.CHANNELS,
            rate=RATE,
            input_device_index=fb_micPort, # loose mic
            input=True,
            stream_callback=self.callback
        )


    def mics_align(self, mic1_arr, mic2_arr):
        assert (mic1_arr.size > 24000 and mic2_arr.size > 24000)
        min = 65535
        best_shift = -2000
        slice = mic1_arr[2000:mic1_arr.size-2000]
        for shift in range(-2000, 2000):
            check = mean_squared_error(slice, mic2_arr[2000+shift:mic2_arr.size-2000+shift])
            if (check < min):
                min=check
                best_shift = shift
        if (best_shift >= 0):
            mic2_arr = mic2_arr[abs(best_shift):]
            mic1_arr = mic1_arr[:mic1_arr.size-abs(best_shift)]
        elif (best_shift < 0):
            mic1_arr = mic1_arr[abs(best_shift):]
            mic2_arr = mic2_arr[:mic2_arr.size-abs(best_shift)]

        print("The best shift was: ", best_shift)
        print("The resulting mse was: ", min)
        return mic1_arr, mic2_arr


    def callback(self, in_data, frame_count, time_info, status):
        noise = np.frombuffer(in_data, dtype=np.int16)

        if (time_info['input_buffer_adc_time'] != 0):
            try:
                self.mics[0][self.index0:self.index0+noise.size*self.CHANNELS] = noise
                self.index0 = self.index0+noise.size*self.CHANNELS
            except:
                self.index0 = self.index0+noise.size*self.CHANNELS

        else:
            try:
                self.mics[1][self.index1:self.index1+noise.size*self.CHANNELS] = noise
                self.index1 = self.index1+noise.size*self.CHANNELS
            except:
                self.index1 = self.index1+noise.size*self.CHANNELS

        if(self.index0 > self.noise_len and self.index1 > self.noise_len ):
            return (in_data, pyaudio.paAbort)
        else:
            return (in_data, pyaudio.paContinue)

    def listen(self):
        ext_mic = np.zeros(self.noise_len)
        des_mic = np.zeros(self.noise_len)
        self.ff_mic.start_stream()
        self.fb_mic.start_stream()
        print('\nI\'m Listening...')

        while self.ff_mic.is_active():
            time.sleep(0.1)

        self.ff_mic.close()
        self.fb_mic.close()
        print("Stream stopped")
        self.ff.terminate()
        self.fb.terminate()

        # Getting rid of blip and plotting the delay
        stop_idx =-1
        for i in range(self.mics[0].size-1, -1, -1):
            if (self.mics[0][i] != 0):
                stop_idx = i
                print('index',stop_idx)
                break

        ext_mic = self.mics[0][50000:stop_idx+1]
        des_mic = self.mics[1][50000:stop_idx+1]
        
        # plt.figure(figsize=(6.29,2.36))
        # plt.plot(ext_mic)
        # plt.plot(des_mic)
        # plt.show()
    #Fetching correct noisy files
    # ext_noisy = wf.read('sounds/mic_profiles/sennheiser.wav')
    # des_noisy = wf.read('sounds/mic_profiles/bberry.wav')
    # #Removing static noise from mic signals
    # ext_mic = nr.reduce_noise(audio_clip=ext_mic, noisy_clip=ext_noisy, verbose=True)
    # des_mic = nr.reduce_noise(audio_clip=des_mic, noisy_clip=des_noisy, verbose=True)

    # plt.plot(ext_mic, label='Ext. mic')
    # plt.plot(des_mic, alpha=0.8, label='Int. mic')

        ext_mic, des_mic = self.mics_align(ext_mic, des_mic) # Fixing delay
        # print(ext_mic.size)
        # print(des_mic.size)
        # print(ext_mic.size-des_mic.size)
        ff_micFile = 'sounds/ff_mic.wav'
        fb_micFile = 'sounds/fb_mic.wav'
        wf.write(ff_micFile, 44100, ext_mic)
        wf.write(fb_micFile, 44100, des_mic)

        return ff_micFile, fb_micFile
