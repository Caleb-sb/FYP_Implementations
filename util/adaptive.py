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
rp = ReportPlot(title="Error as Filter Approaches Ideal", xlabel="Samples", ylabel="RMSE", size=14, ticksize=7)
plt.figure(figsize = rp.figsize)
fc = 0.03 #Resulting in crit of fc*44100
N=51
n = np.arange(N)
sinc_func = np.sinc(2*fc*(n-(N/2)))
h_window  = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
sinc_func = sinc_func * h_window
sinc_func = sinc_func / np.sum(sinc_func)
ideal_filt = sinc_func
#-----------------------------------------------------------------------------#
def LMS_filter(taps, u, d, music, h = np.zeros(50)):
    assert (np.max(u) <= 1 and np.max(d) <= 1), "Data should be normalised!"
    y = np.zeros(len(u)-taps+1)
    e = np.zeros(len(u)-taps+1)
    fe= np.zeros(len(u)-taps+1) # FOR ATP2
    for n in range(len(u)-taps+1):
            x = np.flipud(u[n:n+taps])  # Slice to get view of 'taps' latest datapoints
            y[n] = -1*np.dot(x, h) #+ music[n]
            e[n] = d[n+taps-1] + y[n] #- music[n]
            #-----------------------------------------------------------------
            if taps == 51 and n>0:
                fe[n]= np.sqrt(mse(h,ideal_filt)) # FOR THE ACCEPTANCE TEST
            #-----------------------------------------------------------------
            h = h + 1/(600) * x * e[n]
            #y[n] = -1*np.dot(x, h) #+ music[n]
    # -------------------------------------------------------------------------
    if taps == 51:
        rp.plotPy(np.linspace(0, y.size-1, y.size-1, dtype=np.int32), fe[1:], label='LMS')
    # -------------------------------------------------------------------------
    return y, e, h

def NLMS_filter(taps, u, d, music, h = np.zeros(50)):
    assert (np.max(u) <= 1 and np.max(d) <= 1), "Data should be normalised!"
    y = np.zeros(len(u)-taps+1)
    e = np.zeros(len(u)-taps+1)
    fe= np.zeros(len(u)-taps+1) # FOR THE ACCEPTANCE TEST
    for n in range(len(u)-taps+1):
            x = np.flipud(u[n:n+taps])  # Slice to get view of 'taps' latest datapoints
            y[n] = -1*np.dot(x, h) #+ music[n]
            e[n] = d[n+taps-1] + y[n] #- music[n]
            if taps == 51 and n>0:
                fe[n]= np.sqrt(mse(h,ideal_filt)) # FOR THE ACCEPTANCE TEST
            h = h + 1/(np.dot(x,x)) * x * e[n]
    #        y[n] = -1*np.dot(x, h) #+ music[n]
    if taps == 51:
        print("hello")
        rp.plotPy(np.linspace(0, y.size-1, y.size-1, dtype=np.int32), fe[1:], label='NLMS')
        plt.legend(loc='upper right')
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
