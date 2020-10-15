import sys
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from versions.v1 import SimANC
from versions.v2 import MicsInput
from util.reportable import ReportPlot
import time

def calc_mse(item, heard_ambi):
    h_ambi = np.array(heard_ambi[item.order-1:item.y.size+(item.order-1)])
    error_squared = (h_ambi + item.y)**2
    print(item.order)
    if(item.order == 50 and item.mode=='nlms'):
        rp = ReportPlot(title="NLMS Error", xlabel="Samples", ylabel="Amplitude", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), item.y, label='Output')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi, label='Interference')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi+item.y, label='Error', color='black')
        plt.legend(loc='upper right')
        wf.write('sounds/nlms_cancelling_sig.wav', 44100, item.y)
        print(item.h)
        for idx in range(item.y.size):
            if (all(i <= 0.01 for i in abs((item.y+h_ambi)[idx:]))):
                print(idx, "NLMS Here")
                break
    return (np.sqrt(np.mean(error_squared)))


def power_reduction(item, heard_ambi):
    h_ambi = np.array(heard_ambi[item.order-1:item.y.size+(item.order-1)]) #Have to change the alignment individually for a fair comparison between orders

    reference_power = np.dot(h_ambi, h_ambi)
    resultant = h_ambi + item.y

    power = np.dot(resultant, resultant)
    gain = 10*np.log(power/reference_power)
    print(item.order, gain)
    if(item.order == 50 and item.mode=='lms'):
        rp = ReportPlot(title="LMS Error", xlabel="Samples", ylabel="Amplitude", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), item.y, label='Output')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi, label='Interference')
        rp.plotPy(np.linspace(0, item.e.size, item.e.size, dtype=np.int32), h_ambi+item.y, label='Error', color='black')
        plt.legend(loc='upper right')
        # for idx in range(item.y.size):
        #     if (all(i <= 0.01 for i in abs((item.y+h_ambi)[idx:]))):
        #         print(idx, "LMS Here")
        #         break
    return gain

def test(arr, heard_ambi):
    out_errors = []
    out_gain = []
    for test_items in arr:
        for item in test_items:
            if item.mode == 'fxlms':
                s_path = np.zeros(50)
                s_path[0] = 1
                item.adapt(s_path)
            else:
                item.adapt()
                print(item.mode, item.order)
                # print(np.sum(np.abs(item.e)))


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
        rp.plotPy(np.linspace(2, 50, 5, dtype=np.int8), out_errors[0], 'LMS')
        rp.plotPy(np.linspace(2, 50, 5, dtype=np.int8), out_errors[1], 'NLMS')
        plt.legend(loc='upper right')

        rp = ReportPlot(title="Signal Power Reduction vs Taps", xlabel="Number of Taps", ylabel="Gain (dB)", size=14, ticksize=7)
        plt.figure(figsize  = rp.figsize)
        rp.plotPy(np.linspace(2, 50, 5, dtype=np.int8), out_gains[0], 'LMS')
        rp.plotPy(np.linspace(2, 50, 5, dtype=np.int8), out_gains[1], 'NLMS')
        plt.legend(loc='center right')
    except (ValueError):
        print('Results are shortened. Uncomment line 90 in testing.py for more')

def version1(ref_fname = 'sounds/short_ambi_noise.wav', err_fname = 'sounds/heard_ambi.wav', ideal_needed=True):
    ideal_taps  =   50
    orders      =   np.linspace(50, 2*ideal_taps, 5, dtype=np.int8)
    # orders      =   np.array([50])

    # Producing the ideal 'acoustic' filter
    if (ideal_needed):
        # Fetching the reference noise for ideal acoustic filtering
        sr, ambi = wf.read(ref_fname, True)
        ambi = np.array(ambi, dtype=np.float32)
        ambi = ambi/np.max(ambi)

        H = np.zeros(ideal_taps)
        H[0:13] = np.array([0.5+0.5*np.cos((4*np.pi*n)/ideal_taps) for n in range(0,13)])
        H[37:]  = np.array([0.5-0.5*np.cos((4*np.pi*n)/ideal_taps) for n in range(0,13)])
        shift = np.zeros(ideal_taps)
        shift[24] = 1
        H = H*np.fft.fft(shift)
        h = np.real(np.fft.ifft(H))

        heard_ambi = np.real(np.convolve(h, ambi, mode ='full'))
        wf.write(err_fname, sr, heard_ambi)
    else:
        sr, heard_ambi = wf.read(err_fname, True)
        heard_ambi = np.array(heard_ambi, dtype=np.float32)
        # heard_ambi = heard_ambi/np.max(heard_ambi)

    print(heard_ambi.size, np.mean(abs(heard_ambi)))

    # Initing v1s for testing
    lms_test_items  = np.array([SimANC(
                            order=order,
                            mode='lms',
                            ref_fname = ref_fname,
                            err_fname = err_fname)
                            for order in orders])
    nlms_test_items = np.array([SimANC(
                            order=order,
                            mode='nlms',
                            ref_fname = ref_fname,
                            err_fname = err_fname)
                            for order in orders])

    # Loads each item with the cancelling signal, error and estimated path
    test(np.array([lms_test_items, nlms_test_items]), heard_ambi)
    plt.show()

def version2():
    inport = 21 #Merged the devices using .asoundrc
    duration = 15
    mi = MicsInput(duration=duration, inport=inport)
    # print('helllllllllllllllllllllllllllllllllllllllllllllll')
    # time.sleep(10)
    # print("hello")
    ff_mic, fb_mic = mi.listen()
    version1(ref_fname = ff_mic, err_fname= fb_mic, ideal_needed=False)


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
