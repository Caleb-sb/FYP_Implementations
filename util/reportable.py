import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager as fm, rcParams
import matplotlib
"""
Wrapper for pyplot to make consistently nice graphs for the report
"""
class ReportPlot:
    plt.style.use('ggplot')

    figsize = (6.69,2.76)

    fpath = os.path.join(rcParams["datapath"], "fonts/ttf/LMRoman12-Regular.ttf")
    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]

    def __init__(this, title="", xlabel="", ylabel="", size=14, ticksize=7):
        this.title = title
        this.xlabel = xlabel
        this.ylabel = ylabel
        this.size   = size
        this.ticksize = ticksize

    def plotPy(this, x, y, label, color='', alpha = 1):
        rcParams['axes.facecolor'] = 'white'
        matplotlib.rc('xtick', labelsize=this.ticksize)
        matplotlib.rc('ytick', labelsize=this.ticksize)

        # plt.figure(figsize=this.figsize)
        plt.title(this.title, fontproperties=this.prop, size=this.size)
        plt.xlabel(this.xlabel, fontproperties=this.prop, size=this.size)
        plt.ylabel(this.ylabel, fontproperties=this.prop, size=this.size)
        if (color == '' and alpha == 1):
            plt.plot(x, y, label=label)
        else:
            plt.plot(x, y, label=label, color=color, alpha=alpha)
