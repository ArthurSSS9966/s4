import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce
from tensorpac import Pac


class Data_Class:
    def __init__(self, y, T, seq_id, trial_id):
        """

        :param y:
        :param T:
        :param seq_id:
        :param trial_id:
        """
        self.y = y
        self.T = T
        self.trial_id = trial_id
        self.xsm = None
        self.Vsm = None
        self.VsmGP = None
        self.seq_id = seq_id

        def __repr__(self):
            return "(Trial id: %d, T: %d, seq id: %d,x: %s, y: %s)" % (
                self.trial_id,
                self.T,
                self.seq_id,
                np.array_repr(self.x),
                np.array_repr(self.y),
            )


class Kfclass:
    def __init__(self, mu_1, V1, A, Q, C, d, R, curGain):
        self.mu_1 = mu_1
        self.V1 = V1
        self.A = A
        self.Q = Q
        self.C = C
        self.d = d
        self.R = R
        self.curGain = curGain

        def __repr__(self):
            return "(mui_1: %s, V1: %s, A: %s,Q: %s, C: %s,d: %s,R: %s)" % (
                np.array_repr(self.mu_1),
                np.array_repr(self.V1),
                np.array_repr(self.A),
                np.array_repr(self.Q),
                np.array_repr(self.C),
                np.array_repr(self.d),
                np.array_repr(self.R),
            )


class Mmatclass:
    def __init__(self, M0, M1, M2, miu1):
        self.M0 = M0
        self.M1 = M1
        self.M2 = M2
        self.mu1 = miu1

        def __repr__(self):
            return "(M0: %s, M1:%s, M2:%s)" % (
                np.array_repr(self.M0),
                np.array_repr(self.M1),
                np.array_repr(self.M2),
            )


def normalize(x):
    if x.ndim == 1:
        return x
    for i in range(0, len(x)):
        tep = x[i, :]
        x[i, :] = (tep - min(tep)) / (max(tep) - min(tep))
    return x


class PSDclass:
    alphapsd = []
    betapsd = []
    gammapsd = []
    highgammapsd = []
    movedata = []
    rawdata = []
    angle = []

    def __init__(
        self,
        trial=None,
        seqid=None,
        phase=None,
        f=None,
        t=None,
        Sxx=None,
        TSxx=None,
        tt=None,
        ft=None,
        trialtime=None,
    ):
        self.success = None
        self.trial: int = trial
        self.phase = phase
        self.seqid = seqid
        self.f = f
        self.t = t
        self.Sxx = Sxx
        self.Tsxx = TSxx
        self.tt = tt
        self.ft = ft
        self.trialtime = trialtime
        # #Define a PAC object
        # self.p = Pac(idpac=(6, 0, 0), f_pha='hres', f_amp='hres')

    def addalpha(self, psd):
        self.alphapsd = psd

    def addbeta(self, psd):
        self.betapsd = psd

    def addgamma(self, psd):
        self.gammapsd = psd

    def addhgamma(self, psd):
        self.highgammapsd = psd

    def addmove(self, movedata):
        self.movedata = movedata

    def addraw(self, raw):
        self.rawdata = raw

    def addangle(self, angle):
        self.angle = angle

    def addalphaspectrogram(self, psd):
        self.alphaspec = psd

    def addbetaspectrogram(self, psd):
        self.betaspec = psd

    def addgammaspectrogram(self, psd):
        self.gammaspec = psd

    def addhgammaspectrogram(self, psd):
        self.highgammaspec = psd

    def getband(self, band, datatype="psd"):
        if datatype == "psd":
            if band == "alpha":
                return self.alphapsd
            elif band == "beta":
                return self.betapsd
            elif band == "gamma":
                return self.gammapsd
            else:
                return self.highgammapsd
        elif datatype == "spec":
            if band == "alpha":
                return self.alphaspec
            elif band == "beta":
                return self.betaspec
            elif band == "gamma":
                return self.gammaspec
            else:
                return self.highgammaspec

    def get_raw(self):
        return self.rawdata

    def getRMS(self):
        return self.movedata

    def add_success(self, success):
        self.success = success


class Reaching_class:
    def __init__(self, phase, band, trial=[], seqid=[]):
        self.trialnumber = trial
        self.seqid = seqid
        self.phases = phase
        self.bands = band
        self.data = {
            phase: {band: [] for band in self.bands} for phase in self.phases
        }

    def add_data(self, phase, band, value):
        self.data[phase][band].append(value)


class compressedpsd:
    """ """

    psdtep = []

    def __init__(self, phaseseq):
        self.phase = phaseseq
        self.psdtep = [None] * len(self.phase)

    def add_phase_data(self, psddata: PSDclass, band):
        localphse = psddata.phase
        storeindex = self.phase.index(localphse)
