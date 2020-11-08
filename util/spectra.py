import numpy as np
from util.reportable import ReportPlot
import matplotlib.pyplot as plt

def plot_dbspectrum(x, fs=44100, title="Spectrum"):
    N = x.size
    if (N%2==0):
        half = (N/2) +1
    else:
        half = N/2
    spectrum = abs(np.fft.rfft(x))
    freq = np.arange(half) / (float(N) / fs)
    s_mag = spectrum*2 / N
    deci_spec = 20*np.log10(s_mag/1)
    # deci_spec = spectrum
    deci_spec = np.array([max(-99, i) for i in deci_spec])
    rp = ReportPlot(title=title, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", size=14, ticksize=7)
    plt.figure(figsize  = rp.figsize)
    plt.ylim(max(-95, np.min(deci_spec)), np.max(deci_spec)+5)
    plt.xlim(0, 5500)
    rp.plotSpectrum(freq, deci_spec, label="ambi")
