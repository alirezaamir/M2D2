import os
import sys
import tables
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import signal, integrate
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, \
     linear_kernel, polynomial_kernel, manhattan_distances

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 512
BANDS = [(1,4), (4,8), (8,12), (12,30)]

def main():
    kernel_name = "rbf"
    if len(sys.argv) > 1:
        kernel_name = sys.argv[1]

    if kernel_name == "rbf":
        kernel = rbf_kernel
    elif kernel_name == "laplacian":
        kernel = laplacian_kernel
    elif kernel_name == "linear":
        kernel = linear_kernel
    elif kernel_name == "polynomial":
        kernel = polynomial_kernel
    else:
        raise NotImplementedError("Kernel: {} is invalid".format(kernel_name))
    
    LOG.info("Using kernel: {}".format(kernel_name))
    
    dirname = "../temp/psd/{}".format(kernel_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            LOG.info("Processing: {}".format(node._v_name))
            if len(node.attrs.seizures) != 1:
                continue
            
            data = node.read()
            seizures = node.attrs.seizures
            X, y = data[:,:-1], data[:,-1]
            start = np.min(np.where(y > 0)[0])
            stop = np.max(np.where(y > 0)[0])

            buff_mins = 20
            minv = max(0, start-(buff_mins*60*SF))
            maxv = min(X.shape[0], stop+(buff_mins*60*SF))
            data = data[minv:maxv,:]
            X = X[minv:maxv,:]
            y = y[minv:maxv]
            
            sos = signal.butter(3, 50, fs=SF, btype="lowpass", output="sos")
            X = signal.sosfilt(sos, X, axis=1)
            Z = []
            q = []
            for ix in range(SEG_LENGTH, X.shape[0], SEG_LENGTH):
                Z.append(compute_band_relpower(X[ix-SEG_LENGTH:ix,:]))
                q.append(np.any(y[ix-SEG_LENGTH:ix]))
            
            Z = np.vstack(Z)
            y = np.array(q)
            band_names = ["{}-{}".format(x,y) for x,y in BANDS]
            colnames = [(x + "_1", x + "_2") for x in band_names]
        
            bands = pd.DataFrame(Z, columns=[n for x in colnames for n in x])
            plt.close()
            bands.plot()
            plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")            
            plt.legend()
            plt.savefig("{}/{}_signal_relpower.png".format(dirname, node._v_name))

            K = kernel(Z)
            mmd = []
            for N in range(1,Z.shape[0]):
                M = Z.shape[0] - N
                Kxx = K[:N,:N].sum()
                Kxy = K[:N,N:].sum()
                Kyy = K[N:,N:].sum()
                mmd.append(np.sqrt(
                    ((1/float(N*N))*Kxx) + 
                    ((1/float(M*M))*Kyy) -
                    ((2/float(N*M))*Kxy)
                ))
            
            ws = []
            mmd = np.array(mmd)
            mmd_corr = np.zeros(mmd.size)
            for ix in range(1,mmd_corr.size):
                w = ((Z.shape[0]-1) / float(ix*(N-ix)))
                ws.append(w)
                mmd_corr[ix] = mmd[ix] - w*mmd.max()

            # mmd = mmd[200:-200]
            # mmd_corr = mmd_corr[200:-200]
            plt.close()
            plt.plot(mmd, label="MMD")
            plt.plot(mmd_corr, label="MMD (Corrected)")
            plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.savefig("{}/{}_mmd.png".format(dirname, node._v_name))


def compute_band_relpower(X):
    freqs, psd = signal.welch(X, SF, axis=0)
    freq_res = freqs[1] - freqs[0]
    total_power = integrate.simps(psd, dx=freq_res, axis=0)

    where = total_power <= 1e-5
    total_power[where] = -1

    band_relpower = []
    for lb, ub in BANDS:
        idx = np.logical_and(freqs >= lb, freqs < ub)
        band_power = integrate.simps(psd[idx,:], dx=freq_res, axis=0)
        relpow = band_power / total_power
        relpow[where] = 0
        band_relpower.append(relpow)
    
    return np.concatenate(band_relpower)


if __name__=="__main__":
    main()
