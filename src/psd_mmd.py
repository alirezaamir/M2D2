import tables

import numpy as np

from scipy import signal, integrate
from sklearn.metrics.pairwise import rbf_kernel

SF = 256
SEG_LENGTH = 1024
BANDS = [(1,4), (4,8), (8,12), (12,30)]

def main():
    X_train, y_train = [], []

    test_subject = "chb01"
    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        # first figure out the number of non-seizure windows to sample
        for node in h5_file.walk_nodes("/{}".format(test_subject), "CArray"):
            data = node.read()
            if len(node.attrs.seizures) > 0:
                continue
            else:
                ixs = np.random.random_integers(SEG_LENGTH, data.shape[0], size=5000)
            X = compute_band_relpower([data[ix-SEG_LENGTH:ix,:-1] for ix in ixs])
            y = [np.any(data[ix-SEG_LENGTH:ix,-1] > 0) for ix in ixs]

            X_train.append(X)
            y_train.append(y)
    
    X_train = np.concatenate(X_train)

    y_test = []
    X_test = []
    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        # first figure out the number of non-seizure windows to sample
        for node in h5_file.walk_nodes("/{}".format(test_subject), "CArray"):
            if len(node.attrs.seizures) == 0:
                continue
            data = node.read()
            X = []
            for ix in range(SEG_LENGTH, data.shape[0]):
                X.append(compute_band_relpower(
                    np.expand_dims(data[ix-SEG_LENGTH:ix,:-1], axis=0)
                    ).ravel())
            X_test.append(np.vstack(X))
            break

    
    X_test = np.concatenate(X_test)
    ixs = range(512, X_test.shape[0])
    mmd = [compute_mmd(X_train, X_test[ix-512:ix,:]) for ix in ixs]


def estimate_prob_mc(X_no, X_yes, Q, mc_iter=30):
    P_no = (1/float(X_no.shape[0]))*rbf_kernel(X_no, Q).sum()
    P_yes = (1/float(X_yes.shape[0]))*rbf_kernel(X_yes, Q).sum()
    return P_yes, P_no


def compute_mmd_mc(X, Y, mc=100):
    N = np.float(X.shape[0])
    M = np.float(Y.shape[0])
    mmd = 0
    for _ in range(mc):
        s = np.random.choice(N, size=1024)
        Kxx = rbf_kernel(X[s,:], X[s,:]).sum()
        Kxy = rbf_kernel(X[s,:], Y).sum()
        Kyy = rbf_kernel(Y, Y).sum()
        mmd += (1/(N*N))*Kxx + (1/(M*M))*Kyy - (2/(N*M))*Kxy
    return mmd / float(mc)


def compute_band_relpower(X):
    freqs, psd = signal.welch(X, SF, axis=1)
    freq_res = freqs[1] - freqs[0]
    total_power = integrate.simps(psd, dx=freq_res, axis=1)

    where = total_power <= 1e-5
    total_power[where] = -1

    band_relpower = []
    for lb, ub in BANDS:
        idx = np.logical_and(freqs >= lb, freqs < ub)
        band_power = integrate.simps(psd[:,idx,:], dx=freq_res, axis=1)
        relpow = band_power / total_power
        relpow[where] = 0
        band_relpower.append(relpow)
    
    return np.concatenate(band_relpower, axis=1)