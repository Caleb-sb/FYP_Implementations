import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from util.reportable import ReportPlot
"""
This file contains the functions used for all iterations of the project:
LMS, NLMS, FXLMS and RLS.
Usage:
    af = AdaptiveFilter(taps, step_size)
    This will set the filter length and the step size for LMS filtering.
"""
# FOR THE ACCEPTANCE TEST
# rp = ReportPlot(title="Error as Filter Approaches Ideal", xlabel="Samples", ylabel="RMSE", size=14, ticksize=7)
# plt.figure(figsize = rp.figsize)
H = np.zeros(50)
H[0:13] = np.array([0.5+0.5*np.cos((4*np.pi*n)/50) for n in range(0,13)])
H[37:]  = np.array([0.5-0.5*np.cos((4*np.pi*n)/50) for n in range(0,13)])
shift = np.zeros(50)
shift[24] = 1
H = H*np.fft.fft(shift)
ideal_filt = np.real(np.fft.ifft(H))
#-----------------------------------------------------------------------------#
def LMS_filter(taps, u, d, music):
    print(len(u), len(d))
    h = np.zeros(taps)
    y = np.zeros(len(u)-taps+1)
    e = np.zeros(len(u)-taps+1)
    fe= np.zeros(len(u)-taps+1) # FOR THE ACCEPTANCE TEST
    for n in range(len(u)-taps+1):
            x = np.flipud(u[n:n+taps])  # Slice to get view of 'taps' latest datapoints
            y[n] = np.dot(x, h) #+ music[n]
            e[n] = d[n+taps-1] - y[n] #- music[n]
            #-----------------------------------------------------------------
            if taps == 50 and n>0:
                fe[n]= np.sqrt(mse(h,ideal_filt)) # FOR THE ACCEPTANCE TEST
            #-----------------------------------------------------------------
            h = h + 1/(13) * x * e[n]
            y[n] = -1*np.dot(x, h) #+ music[n]
    # -------------------------------------------------------------------------
    # if taps == 50:
        # rp.plotPy(np.linspace(0, y.size-1, y.size-1, dtype=np.int32), fe[1:], label='LMS')
    # -------------------------------------------------------------------------
    return y, e, h

def NLMS_filter(taps, u, d, music):
    assert (np.max(u) <= 1 and np.max(d) <= 1), "Data should be normalised!"
    print('MAXES:', np.max(u), np.max(d))
    h = np.zeros(taps)
    y = np.zeros(len(u)-taps+1)
    e = np.zeros(len(u)-taps+1)
    fe= np.zeros(len(u)-taps+1) # FOR THE ACCEPTANCE TEST
    for n in range(len(u)-taps+1):
            x = np.flipud(u[n:n+taps])  # Slice to get view of 'taps' latest datapoints
            y[n] = np.dot(x, h) #+ music[n]
            e[n] = d[n+taps-1] - y[n] #- music[n]
            if taps == 50 and n>0:
                fe[n]= np.sqrt(mse(h,ideal_filt)) # FOR THE ACCEPTANCE TEST
            h = h + 1/(np.dot(x,x)) * x * e[n]
            y[n] = -1*np.dot(x, h) #+ music[n]
    # if taps == 50:
    #     rp.plotPy(np.linspace(0, y.size-1, y.size-1, dtype=np.int32), fe[1:], label='NLMS')
    #     plt.legend(loc='upper right')
    return y, e, h

def FXLMS_filter(taps, u, d, s_path, music):
    assert (np.max(u) <= 1 and np.max(d) <= 1), "Data should be normalised!"

    u = np.convolve(u, s_path, mode='same')
    h = np.zeros(taps)
    y = np.zeros(len(u)-taps+1)
    e = np.zeros(len(u)-taps+1)
    for n in range(len(u)-taps+1):
            x = np.flipud(u[n:n+taps])  # Slice to get view of 'taps' latest datapoints
            y[n] = np.dot(x, h) #+ music[n]
            e[n] = d[n+taps-1] - y[n] #- music[n]
            h = h + 1/(np.dot(x,x)) * x * e[n]
            y[n] = -1*np.dot(x, h) #+ music[n]
    return y, e, h
