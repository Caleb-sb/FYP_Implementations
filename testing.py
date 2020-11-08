import sys
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from versions.v1 import Version1
from versions.v2 import Version2
from util.reportable import ReportPlot
from util.spectra import *

def calc_mse(item, heard_ambi):
    h_ambi = np.array(heard_ambi[item.order-1:item.y.size+(item.order-1)])
    error_squared = (h_ambi + item.y)**2
    print(item.order)
    if(item.order == 51 and item.mode=='nlms'):
        rp = ReportPlot(title="NLMS Output and Interference Signal", xlabel="Samples", ylabel="Amplitude", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), item.y/np.max(item.y), label='Output')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi/np.max(h_ambi), label='Interference')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi/np.max(h_ambi)+item.y/np.max(item.y), label='Error', color='black')
        plt.legend(loc='upper right')
        wf.write('sounds/nlms_cancelling_sig.wav', 44100, item.y)
    return (np.sqrt(np.mean(error_squared)))


def power_reduction(item, heard_ambi):
    h_ambi = np.array(heard_ambi[item.order-1:item.y.size+(item.order-1)]) #Have to change the alignment individually for a fair comparison between orders

    reference_power = np.dot(h_ambi, h_ambi)
    resultant = h_ambi + item.y

    power = np.dot(resultant, resultant)
    gain = 10*np.log(power/reference_power)
    print(item.order, gain)
    if(item.order == 51 and item.mode=='lms'):
        rp = ReportPlot(title="LMS Output and Interference Signal", xlabel="Samples", ylabel="Amplitude", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), item.y/np.max(item.y), label='Output')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi/np.max(h_ambi), label='Interference')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi/np.max(h_ambi)+item.y/np.max(item.y), label='Error', color='black')
        plt.legend(loc='upper right')
    return gain

def test(arr, heard_ambi):
    out_errors = []
    out_gain = []
    for test_items in arr:
        for item in test_items:
            item.adapt()
            print('Adaptation complete for mode: %s, of order: %d'%(item.mode, item.order))

    out_errors = np.array([[calc_mse(item, heard_ambi) for item in test_items] for test_items in arr])
    # rp = ReportPlot(title="Cancelling Signal and Heard Interference", xlabel="Samples", ylabel="Amplitude", size=14, ticksize=7)
    # plt.figure(figsize  = rp.figsize)
    out_gains = np.array([[power_reduction(item, heard_ambi) for item in test_items] for test_items in arr])
    # plt.legend(loc='upper right')
    plt.show()
    print(out_errors)
    print(out_gains)
    try:
        rp = ReportPlot(title="RMSE vs Filter Length", xlabel="Number of Taps", ylabel="RMSE", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(2, 1.5*51, 75, dtype=np.int8), out_errors[0], 'LMS')
        rp.plotPy(np.linspace(2, 1.5*51, 75, dtype=np.int8), out_errors[1], 'NLMS')
        plt.legend(loc='upper right')

        rp = ReportPlot(title="Signal Power Reduction vs Taps", xlabel="Number of Taps", ylabel="Gain (dB)", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(2, 1.5*51, 75, dtype=np.int8), out_gains[0], 'LMS')
        rp.plotPy(np.linspace(2, 1.5*51, 75, dtype=np.int8), out_gains[1], 'NLMS')
        plt.legend(loc='upper right')
    except (ValueError):
        print('Results are shortened. Uncomment line 90 in testing.py for more')

def version1(ref_fname = 'sounds/short_ambi_noise.wav', err_fname = 'sounds/heard_ambi.wav', ideal_needed=True):
    ideal_taps  =   51
    orders      =   np.linspace(2, 1.5*ideal_taps, 75, dtype=np.int8)
    # orders      =   np.array([51])

#---------------------------- To be run for iteration 1 -----------------------#
    if (ideal_needed):
        # Fetching the reference noise for ideal acoustic filtering
        sr, ambi = wf.read(ref_fname, True)
        ambi = np.array(ambi, dtype=np.float32)
        ambi = ambi/np.max(ambi)
        plot_dbspectrum(ambi, title="Spectrum of the Ambient Noise")

        #-------------------- Defining Ideal Lowpass Filter -------------------#

        fc = 0.03 #Resulting in crit of fc*44100
        N=51
        n = np.arange(N)
        sinc_func = np.sinc(2*fc*(n-(N/2)))
        h_window  = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
        sinc_func = sinc_func * h_window
        sinc_func = sinc_func / np.sum(sinc_func)
        h = sinc_func
        plot_dbspectrum(h, title="Lowpass Frequency Response")

        # Filtering the reference noise as the estimation of the primary path
        heard_ambi = np.real(np.convolve(h, ambi, mode ='full'))
        plot_dbspectrum(heard_ambi, title="Spectrum of Heard Ambient Noise")
        # Writing the heard noise to a wav file to be fetched by Version 1
        wf.write(err_fname, sr, heard_ambi)

    #------------------------- To be run for iteration 2 ----------------------#
    else:
        sr, heard_ambi = wf.read(err_fname, True)
        heard_ambi = np.array(heard_ambi, dtype=np.float32)
        # heard_ambi = heard_ambi/np.max(heard_ambi)

    print('Size of heard ambient noise: ', heard_ambi.size)

    # Initing v1s for testing
    lms_test_items  = np.array([Version1(
                            order=order,
                            mode='lms',
                            ref_fname = ref_fname,
                            err_fname = err_fname)
                            for order in orders])
    nlms_test_items = np.array([Version1(
                            order=order,
                            mode='nlms',
                            ref_fname = ref_fname,
                            err_fname = err_fname)
                            for order in orders])

    # Loads each item with the cancelling signal, error and estimated path
    test(np.array([lms_test_items, nlms_test_items]), heard_ambi)
    plt.show()

def version2():
    fb_micPort = 1
    ff_micPort = 9
    duration = 20
    order = 30
    ff_audio = 'sounds/ff_mic.wav'
    fb_audio = 'sounds/fb_mic.wav'
    iter2 = Version2(ff_micPort=ff_micPort, fb_micPort=fb_micPort, duration = duration)
    iter2.start()
    samplerate, heard_ambi = wf.read(fb_audio)
    # print(heard_ambi.size)
    print(calc_mse(iter2, heard_ambi))
    print(power_reduction(iter2, heard_ambi))
    plt.show()
    # version1(ref_fname = ff_mic, err_fname= fb_mic, ideal_needed=False)
    # version1(ref_fname='sounds/ss_mic.wav', err_fname='sounds/bb_mic.wav', ideal_needed=False)


def version3():
    return 3

def main(iteration=1):
    if (iteration == 1):
        print("\nPerforming tests on iteration 1...")
        version1()
    elif (iteration == 2):
        print("\nPerforming tests on iteration 2...")
        version2()
    elif (iteration == 3):
        print("\nPerforming tests on iteration 3...")
        version3()
    else:
        print("\nERROR: Iteration number is not valid.")


if __name__ == '__main__':
    if (len(sys.argv)>1):
        main(int(sys.argv[1]))
    else:
        print('\nERROR: Please enter a valid iteration number to test')
        exit()
