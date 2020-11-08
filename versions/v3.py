'''
        This version attempts to achieve realtime periodic signal active noise
        cancelling. It is intended for signal channel cancelling signals but
        can be expanded to dual channel relatively easily.

        It should be interfaced with a stereo input soundcard for both the
        reference mic and error mic and can be output to a mono or stereo
        output.

---------------------------------------
        Author: Caleb Bredekamp
        Class:  EEE4022S
        Year:   2020
---------------------------------------
'''

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time

# Constants
format  = pyaudio.paInt16
chans   = 2
rate    = 44100

# Filter attributes
taps = 64
h = np.zeros(taps, dtype = np.float32)
#u = np.zeros(256, dtype=np.float32)
#d = np.zeros(256, dtype=np.float32)

@jit(nopython=False)
def callback(in_data, frame_count, time_info, status):
        taps = 64
        h = np.zeros(taps)
        mics = np.frombuffer(in_data, dtype=numba.int16)
        #print(mics.shape)
#       shortstalk[index:index+int(mics.size/2)] = mics[::2]
#       longstalk[index:index+int(mics.size/2)] =mics[1::2]
#       index = index+int(mics.size/2)

        # Splitting into channels and normalising
        u = mics[::2]
#       print(u)
        u = np.array(u/np.max(abs(u)), dtype=np.float32)
#       print(u)
#       print(np.max(u))
#       print(np.min(u))
        d = mics[1::2]
        dscale = np.max(abs(d))
        d = np.array(d/dscale, dtype=np.float32)
        # Getting control signal (will have zeros)
        y = np.zeros(mics.size, dtype=np.float32)
#       n = np.linspace(0, 1024, 1024)
#       y[::2] = 0.5*np.sin((2*np.pi*n)/256)
#       y[1::2] = 0.5*np.sin((2*np.pi*n)/128)
        for n in range(len(u)-taps+1):
                x    = np.flipud(u[n:n+taps])
                y[2*n] = (x*h).sum()
                e    = d[n+taps-1] - y[2*n]
                h    = h + (1/(x*x).sum()) * x * e
                y[2*n] = -1 * (x*h).sum()
                y[2*n+1] = y[2*n]

        out_data = np.array(y*dscale, dtype=np.float32)
#       print(mics[0:20])
        return (out_data.tobytes(), pyaudio.paContinue)

# PyAudio class instance
p = pyaudio.PyAudio()

# Stream object for I/O
audio_io = p.open(
        format  = format,
        channels= chans,
        rate    = rate,
        input   = True,
        output  = True,
        stream_callback     = callback,
        input_device_index  = 1,
        output_device_index = 0
)

# Initialising
audio_io.start_stream()
print('Stream has begun?')

while audio_io.is_active():
        try:
                time.sleep(0.1)
        except:
                plt.figure()
                plt.title('red input (long stalk)')
                plt.plot(longstalk)
                plt.figure()
                plt.title('blue input (short stalk)')
                plt.plot(shortstalk)
                plt.show()
                # print('Interrrupted by user.')
                exit()
