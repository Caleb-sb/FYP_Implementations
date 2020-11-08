#!/usr/bin/env python3

"""Create a JACK client that copies input audio directly to the outputs.

This is somewhat modeled after the "thru_client.c" example of JACK 2:
http://github.com/jackaudio/jack2/blob/master/example-clients/thru_client.c

If you have a microphone and loudspeakers connected, this might cause an
acoustical feedback!

"""
import sys
import signal
import os
import jack
import threading
import numpy as np
from numba import jit, njit
import numba

class Version3:
    def __init__(self, taps, chunk):
        self.order = taps
        self.chunk = chunk
        self.h     = np.zeros(self.order)
        self.mics  = np.zeros(2*self.chunk, dtype=np.float32)
        self.chans = 2
        self.rate  = 44100

        self.res_u = np.zeros(self.order, dtype=np.float32)
        self.res_d = np.zeros(self.order, dtype=np.float32)
        self.init = True


    def process(self, frames):
        assert len(self.client.inports) == len(self.client.outports)
        assert frames == self.client.blocksize
        for i, o in zip(self.client.inports, self.client.outports):

            self.mics[:] = np.frombuffer(i.get_buffer(), dtype=np.float32)
            #print(self.mics[0])
            if (self.init):
                u = self.mics[::2]
                d = self.mics[1::2]
                self.init = False
            else:
                u = np.append(self.res_u, self.mics[::2])
                d = np.append(self.res_d, self.mics[1::2])

            uscale = np.max((u))
            dscale = np.max((d))
            if (dscale != 0 and uscale != 0):
                #print(u.size)
                #print(u)

                self.h, y = filter(u, d, self.order, self.h)
                self.res_d[:] = np.array(d[d.size-self.order:], dtype=np.float32)
                self.res_u[:] = np.array(u[u.size-self.order:], dtype=np.float32)

                #print(np.max(abs(y)))
                out_data = np.array(y, dtype=np.float32)
                o.get_buffer()[:] = out_data.tobytes()
                #o.get_buffer()[:] = i.get_buffer()
            else:
                print('|')
                o.get_buffer()[:] = i.get_buffer()


    def shutdown(self, status, reason):
        print('JACK shutdown!')
        print('status:', status)
        print('reason:', reason)
        self.event.set()

    def begin(self):
        clientname = 'v3'
        servername = None
        self.client = jack.Client(clientname, servername=servername)
        self.client.set_process_callback(self.process)
        self.client.set_shutdown_callback(self.shutdown)
        if self.client.status.server_started:
            print('JACK server started')
        if self.client.status.name_not_unique:
            print('unique name {0!r} assigned'.format(client.name))

        self.event = threading.Event()

        # create two port pairs
        for number in 1, 2:
            self.client.inports.register('input_{0}'.format(number))
            self.client.outports.register('output_{0}'.format(number))


        with self.client:
        # When entering this with-statement, client.activate() is called.
        # This tells the JACK server that we are ready to roll.
        # Our process() callback will start running now.

            capture = self.client.get_ports(is_physical=True, is_output=True)
            if not capture:
                raise RuntimeError('No physical capture ports')

            for src, dest in zip(capture, self.client.inports):
                self.client.connect(src, dest)

            playback = self.client.get_ports(is_physical=True, is_input=True)
            if not playback:
                raise RuntimeError('No physical playback ports')

            for src, dest in zip(self.client.outports, playback):
                self.client.connect(src, dest)

            print('Press Ctrl+C to stop')
            try:
                self.event.wait()
            except KeyboardInterrupt:
                print('\nInterrupted by user')

#________________________________End of Class______________________________#

@jit(nopython=True)
def filter(u, d, order, h):
    # Preparing output
    chunk = 256
    y = np.zeros(2*chunk, dtype=np.float32)
    # Filtering (This might take too long for a given sample rate)
    out_length = int(chunk/2)-order+1
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

def main():
    v3 = Version3(taps=32, chunk=256)
    v3.begin()

if __name__=='__main__':
    main()
