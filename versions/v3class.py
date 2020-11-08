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
import numpy as np
from numba import jit
import numba
import time

class Version3:

    def __init__(self, taps, chunk):
        self.order = taps
        self.chunk = chunk
        self.h     = np.zeros(self.order)
        self.mics  = np.zeros(2*self.chunk, dtype=np.int16)
        self.format= pyaudio.paInt16
        self.chans = 2
        self.rate  = 44100

        self.res_u = np.zeros(self.order, dtype=np.int16)
        self.res_d = np.zeros(self.order, dtype=np.int16)
        self.init = True

    def callback(self, in_data, frame_count, time_info, status):
        # Converting the input to array
        self.mics[:] = np.frombuffer(in_data, dtype = np.int16)
        if (self.init):
            u = self.mics[::2]
            d = self.mics[1::2]
            self.init = False                                                                                                else:                                                                                                                    u = np.append(self.res_u, self.mics[::2])
            d = np.append(self.res_d, self.mics[1::2])

        uscale = np.max(u)
        u = np.array(u/uscale, dtype=np.float32)
        dscale = np.max(d)
        d = np.array(d/dscale, dtype=np.float32)

        self.h, y = filter(u, d, self.order, self.h)
        self.res_d[:] = np.array(d[d.size-self.order:]*dscale, dtype=np.int16)
        self.res_u[:] = np.array(u[u.size-self.order:]*uscale, dtype=np.int16)

        out_data = np.array(y*dscale, dtype=np.int16)
#        print(self.mics.size)
#        print(type(self.mics[0]))
        # Breaking input down into channels and normalising

        outdata = out_data.tobytes()
        return (outdata, pyaudio.paContinue)

    def begin(self):
        # PyAudio class instance
        self.p = pyaudio.PyAudio()

        # Stream object for I/O
        self.audio_io = self.p.open(
                format  = self.format,
                channels= self.chans,
                rate    = self.rate,
                input   = True,
                output  = True,
                stream_callback     = self.callback,
                input_device_index  = 1,
                output_device_index = 0
        )

        self.audio_io.start_stream()
        print('Stream has begun...')
#________________________________End of Class______________________________#

@jit(nopython=True)
def filter(u, d, order, h):
    # Preparing output
    y = np.zeros(u.size*2, dtype=np.float32)
    # Filtering (This might take too long for a given sample rate)
    out_length = len(u)-order+1
    for n in range(out_length):
        lc = 2*n
        rc = 2*n+1
        x  = np.flipud(u[n:n+order])
        y[lc]   = (x*h).sum()
        e       = d[n+order-1] - y[lc]
        h  = h + (1/(x*x).sum()) * x * e
        y[lc]   = -1 * (x*h).sum()
        y[rc]   = y[lc]
    return h, y

def main():                                                                                                              v3 = Version3(taps=64, chunk=1024)                                                                                   v3.begin()
    while v3.audio_io.is_active():
        try:
                time.sleep(0.1)
        except:
                # print('Interrrupted by user.')
                exit()

if __name__ == '__main__':
    main()
