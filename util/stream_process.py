import numpy as np
from scipy.io import wavfile as wf
from util.adaptive import NLMS_filter
from util.audio_io import AudioBuffer
from multiprocessing import Process

class StreamProcessor:
    def __init__(self, order = 30, ff_micPort=9, fb_micPort=1, ff_filename= 'sounds/ff_mic.wav', fb_filename= 'sounds/fb_mic.wav'):
        self.order       = order
        self.h           = np.zeros(order)
        self.mode        = 'nlms' # Only nlms for this iteration
        self.ff_filename = ff_filename
        self.fb_filename = fb_filename
        self.y = [[0 for _ in range(self.order-1)]]
        self.e = [[0 for _ in range(self.order-1)]]
        self.u = []
        self.d = []


    def process(self, stream_array):
        index = 0

        while(len(self.y) < duration*44100):
                # Save the buffer state in case it changes while processing
                interference[0] = np.array([i for i in output_array[0]])
                interference[1] = np.array([i for i in output_array[1]])
                # Process the buffers
                u_scale = np.max(interference[0])
                d_scale = np.max(interference[1])

                self.u.append([item for item in interference[0]])
                self.d.append([item for item in interference[1]])
                u = np.array(self.u).reshape(np.array(self.u).size)
                d = np.array(self.d).reshape(np.array(self.d).size)


                if (index==0):
                    out, error, self.h = NLMS_filter(
                                    taps = self.order,
                                    u = u/u_scale,
                                    d = d/d_scale,
                                    music=np.zeros(50),
                                    h=self.h)

                else:
                    out, error, self.h = NLMS_filter(
                                    taps = self.order,
                                    u = u[index-self.order+1:u.size]/u_scale,
                                    d = d[index-self.order+1:d.size]/d_scale,
                                    music=np.zeros(50),
                                    h=self.h)
                print("THIS IS OUTPUT SIZE: ", out.size)
                print("This is index: ", index)
                if (index == 0):
                    for i in out:
                        self.y[0].append(i*d_scale)
                    for e in error:
                        self.e[0].append(e)
                else:
                    self.y.append([y*d_scale for y in out])
                    self.e.append([e for e in error])
                index = index + interference[0].size

        audio.terminate()
            #     break
        print("Y Length:", len(self.y))
        for item in self.y:
            print("Item Length: ", len(item))
        self.y = np.array(self.y, dtype=np.int16).reshape((np.array(self.y).size))
        self.y = self.y[self.order-1:]
        self.u = np.array(self.u, dtype=np.int16).reshape((np.array(self.u).size))
        self.d = np.array(self.d, dtype=np.int16).reshape((np.array(self.d).size))
        self.e = np.array(self.e, dtype=np.int16).reshape((np.array(self.e).size))
        self.e = self.e[self.order-1:]
        wf.write(self.ff_filename, 44100, self.u)
        wf.write(self.fb_filename, 44100, self.d)
        wf.write('sounds/cancelling_iter2.wav', 44100, self.y[self.order-1:])
