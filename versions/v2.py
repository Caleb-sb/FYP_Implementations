import pyaudio
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from util.adaptive import NLMS_filter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf


class Version2:
    def __init__(self,ff_micPort, fb_micPort, duration, ff_filename= 'sounds/ff_mic.wav', fb_filename= 'sounds/fb_mic.wav', size = 1024*600, ):
        self.noise_len  = size
        self.order = 30
        self.ff_filename = ff_filename
        self.fb_filename = fb_filename
        self.mics       = np.array([np.zeros(self.noise_len, dtype=np.int16),
                                    np.zeros(self.noise_len, dtype=np.int16)])
        self.to_process = np.array([np.zeros(int(self.noise_len/2), dtype=np.int16),
                                    np.zeros(int(self.noise_len/2), dtype=np.int16)])
        self.duration   = duration
        self.limit      = int(size/2)
        self.ff_micPort = ff_micPort
        self.fb_micPort = fb_micPort
        self.ff         = pyaudio.PyAudio()
        self.fb         = pyaudio.PyAudio()
        self.channels   = 1

        self.w_index0, self.w_index1 = 0,0
        self.r_index0, self.r_index1 = 0,0
        self.p_index0, self.p_index1 = 0,0

        self.h10_ready, self.h11_ready = False, False
        self.h20_ready, self.h21_ready = False, False
        self.ready      = False
        self.delay      = 0
        self.calibrating = 1

        self.mode = 'nlms'
        self.y = [[0 for _ in range(self.order-1)]]
        self.e = [[0 for _ in range(self.order-1)]]
        self.u = np.zeros(900*1024)
        self.d = np.zeros(900*1024)
        self.h = np.zeros(self.order)

    def callback(self, in_data, frame_count, time_info, status):
        noise = np.frombuffer(in_data, dtype=np.int16)
        old_index0 = 0
        old_index1 = 0

        if (time_info['input_buffer_adc_time'] != 0):
            if (self.calibrating ==1 and self.w_index0+noise.size == self.noise_len and self.delay>0):
                self.mics[0][self.w_index0:self.w_index0+noise.size*self.channels] = noise
                self.calibrating =0
                self.mics[0][0:self.noise_len-abs(self.delay)] = self.mics[0][abs(self.delay):]
                self.w_index0 = self.noise_len-abs(self.delay)

            else:
                self.mics[0][np.arange(self.w_index0, self.w_index0+noise.size) % self.noise_len] = noise
                old_index0 = self.w_index0
                self.w_index0 = (self.w_index0+noise.size) % self.noise_len

            if(old_index0 > self.w_index0):
                old_index0 = 0
                self.h20_ready=True
                # print("h20_ready")
            elif(self.w_index0 >= int(self.noise_len/2) and self.h10_ready==False and self.r_index0 != int(self.noise_len/2)):
                self.h10_ready= True
                # print("h10_ready")

        else:

            if (self.calibrating ==1 and self.w_index1+noise.size == self.noise_len and self.delay<0):
                self.mics[1][self.w_index1:self.w_index1+noise.size*self.channels] = noise
                self.calibrating =0
                self.mics[1][0:self.noise_len-abs(self.delay)] = self.mics[1][abs(self.delay):]
                self.w_index1 = self.noise_len-abs(self.delay)

            else:
                # print("OI BISH THIS IS WORKING????????111111111111111111111")
                self.mics[1][np.arange(self.w_index1, self.w_index1+noise.size) % self.noise_len] = noise
                old_index1 = self.w_index1
                # print("THIS IS OLD INDEX1: ", old_index1)
                self.w_index1 = (self.w_index1+noise.size) % self.noise_len
                # print("THIS IS NEW INDEX1: ", self.w_index1)

            if(old_index1 > self.w_index1):
                old_index1 = 0
                self.h21_ready=True
                wf.write('test1.wav', 44100, self.mics[1])
                # print("h21_ready")
            elif(self.w_index1 >= int(self.noise_len/2) and self.h11_ready==False and self.r_index1 != int(self.noise_len/2)):
                self.h11_ready= True
                # print("h11_ready")


            print(self.h10_ready, self.h11_ready, self.h20_ready, self.h21_ready, self.ready)

        return (in_data, pyaudio.paContinue)

    def start(self):
        self.ff_mic = self.ff.open(
            format  =pyaudio.paInt16,
            channels=1,
            rate    =44100,
            input_device_index=self.ff_micPort, # sennheiser
            input=True,
            stream_callback=self.callback
        )
        self.fb_mic = self.fb.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input_device_index=self.fb_micPort, # loose mic
            input=True,
            stream_callback=self.callback
        )
        print("Hello there")
        self.ff_mic.start_stream()
        self.fb_mic.start_stream()

        print('\nI\'m Listening...')
        done = -1
        while self.ff_mic.is_active():

            if self.h10_ready:
                self.to_process[0] = self.mics[0][:int(self.noise_len/2)]
                # print("h10 has been set")
            elif self.h20_ready:
                self.to_process[0] = self.mics[0][int(self.noise_len/2):]
                # print("h20 has been set")
            if self.h11_ready:
                self.to_process[1] = self.mics[1][:int(self.noise_len/2)]
                # print("h11 has been set")
            elif self.h21_ready:
                self.to_process[1] = self.mics[1][int(self.noise_len/2):]
                # print("h21 has been set")
            # print("Still Here")
            if self.h10_ready and self.h11_ready:
                self.ready = True
            elif self.h20_ready and self.h21_ready:
                self.ready = True

            if (self.ready):
                if (self.delay == 0):
                    print("Calibrating microphones for delay")

                    self.align()
                    self.w_index0 = 0
                    self.w_index1 = 0
                    self.ready = False
                    self.h10_ready, self.h11_ready = False, False
                    self.h20_ready, self.h21_ready = False, False
                    print("Alignment Complete")
                elif (self.h10_ready and self.h11_ready):
                    self.r_index0 = int(self.noise_len/2)
                    self.r_index1 = int(self.noise_len/2)
                    self.ready = False
                    self.h10_ready, self.h11_ready = False, False
                    print("First half being processed")
                    done = self.process()
                elif (self.h20_ready and self.h21_ready):
                    self.r_index0 = int(self.noise_len)
                    self.r_index1 = int(self.noise_len)
                    self.h20_ready, self.h21_ready = False, False
                    self.ready = False
                    print("Second half being processed")
                    done = self.process()
            if done == 0:
                break
            else:
                continue
        self.stop()
            #     else:
            #         print("ERRRRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            # time.sleep(0.001)

    def align(self):
        mic1_arr = self.to_process[0][50000:]
        mic2_arr = self.to_process[1][50000:]

        # print(mic2_arr.size)
        assert (mic1_arr.size == mic2_arr.size and mic1_arr.size > 50400)
        min = 65535
        best_shift = -4000
        slice = mic1_arr[4000:50000-4000]
        for shift in range(-4000, 4000):
            check = mean_squared_error(slice, mic2_arr[4000+shift:50000-4000+shift])
            if (check < min):
                min=check
                best_shift = shift
        print("Shift:", best_shift)
        self.delay = best_shift

    def stop(self):
        self.ff_mic.close()
        self.fb_mic.close()
        self.ff.terminate()
        self.fb.terminate()
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

    def process(self):
        # Save the buffer state in case it changes while processing
        # interference[1] = np.array([i for i in self.to_process[1]])
        # Process the buffers
        # u_scale = np.max(self.to_process[0])
        d_scale = np.max(self.to_process[1])

        for i in self.to_process[0]:
            self.u[self.p_index0] = i
            self.p_index0 += 1
        for i in self.to_process[1]:
            self.d[self.p_index1] = i
            self.p_index1 += 1
        # self.d.append([item for item in self.to_process[1]])

        # if (self.p_index==0):
        #     out, error, self.h = NLMS_filter(
        #                     taps = self.order,
        #                     u = u/u_scale,
        #                     d = d/d_scale,
        #                     music=np.zeros(50),
        #                     h=self.h)
        # #
        # else:
        #     out, error, self.h = NLMS_filter(
        #                     taps = self.order,
        #                     u = u[self.p_index-self.order+1:u.size]/u_scale,
        #                     d = d[self.p_index-self.order+1:d.size]/d_scale,
        #                     music=np.zeros(50),
        #                     h=self.h)
        # print("THIS IS OUTPUT SIZE: ", out.size)
        print("This is index: ", self.p_index0)
        # if (self.p_index == 0):
        #     for i in out:
        #         self.y[0].append(i*d_scale)
        #     for e in error:
        #         self.e[0].append(e)
        # else:
        #     self.y.append([y*d_scale for y in out])
        #     self.e.append([e for e in error])
        # self.p_index = self.p_index + self.to_process[1].size
        # u = np.array(self.y, dtype=np.int16).reshape((np.array(self.y).size))
        if (self.p_index0 > self.duration*44100):
            return 0
