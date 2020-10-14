import scipy.io.wavfile as wf
import numpy as np
from util import adaptive

class SimANC:
    """
    --  This is the class containing all functionality of the first iteration.
    --  Its performance will be tested by the testing.py.
    --  It assumes there is a 'sounds' folder in the same directory and
        uses the default names of 'ref_noise.wav' and 'des_noise.wav'
    """
    def __init__(self, order = 50, mode='nlms', ref_fname='sounds/ambi_noise.wav',
        err_fname = 'sounds/heard_ambi.wav'):

        self.mode   = mode

        sr,  ambi   = wf.read(ref_fname, True)
        self.ambi_scale = np.max(ambi)
        self.ambi   = np.array(ambi/np.max(ambi), dtype=np.float32)

        sr2, h_ambi = wf.read(err_fname, True)
        self.h_ambi_scale = np.max(h_ambi)
        self.h_ambi = np.array(h_ambi/np.max(h_ambi), dtype=np.float32)

        self.order = order
        assert sr==sr2, "Samplerates must be the same for both wav files"

    def adapt(self, s_path=np.empty(50)):
        if (self.mode == 'lms'):

            self.y, self.e, self.h = adaptive.LMS_filter(
                                        taps = self.order,
                                        u = self.ambi,
                                        d = self.h_ambi,
                                        music = np.zeros(self.ambi.size))
        elif (self.mode == 'nlms'):
            self.y, self.e, self.h = adaptive.NLMS_filter(
                                        taps = self.order,
                                        u = self.ambi,
                                        d = self.h_ambi,
                                        music = np.zeros(self.ambi.size))
        # elif (self.mode == 'fxlms'):
        #     self.y, self.e, self.h = adaptive.FXLMS_filter(
        #                                 s_path = s_path,
        #                                 taps = self.order,
        #                                 u = self.ambi,
        #                                 d = self.h_ambi,
        #                                 music = np.zeros(self.ambi.size))
        else:
            print("Invalid mode selected!")

        self.y = self.y * self.h_ambi_scale
