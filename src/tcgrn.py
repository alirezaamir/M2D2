import os
import tqdm
import tables
import logging

import numpy as np

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

EIG_THRESH = 1e-2
MAX_EIGS = 32
SF = 256

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def main():
    seg_length = 512
    stride = 256
    h5_file = tables.open_file("../input/eeg_data_temples2.h5")
    for node in h5_file.walk_nodes("/chb10", "CArray"):
        if len(node.attrs.seizures) > 0:
            LOG.info("Processing Node: {}".format(node._v_name))
            process_node(node, seg_length, stride)
            break
    
    h5_file.close()


def process_node(node, seg_length, stride):
    data = node.read()
    S = StandardScaler()

    X = data[:,:-1]
    sos = signal.butter(3, 50, fs=SF, btype="lowpass", output="sos")
    X = signal.sosfilt(sos, X, axis=1)

    X = S.fit_transform(X)
    sites, y = [], []
    for ix in range(seg_length, X.shape[0], stride):
        sites.append(X[ix-seg_length:ix,:])
        y.append(np.any(data[ix-seg_length:ix,-1]))

    if not os.path.exists("../temp/tcgrn"):
        os.makedirs("../temp/tcgrn")

    fh = tables.open_file("../temp/tcgrn/{}.h5".format(node._v_name), mode="w")
    for level in range(1,3):
        LOG.info("Coarse Graining level: {}".format(level))
        level_group = fh.create_group("/", "level{}".format(level))
        
        new_y = []
        new_sites = []
        ixs = list(range(1, len(sites), 2))
        if ixs[-1] != len(sites) - 1:
            ixs.append(len(sites) - 1)
        
        for ix in tqdm.tqdm(ixs):
            site_group = fh.create_group(level_group, "site{}".format(ix))
            L, V, P = compute_eigs(sites[ix], sites[ix-1])
            
            new_y.append(int(y[ix] or y[ix-1]))            
            new_sites.append(P)
            
            fh.create_array(site_group, "eigvects", V)
            fh.create_array(site_group, "eigvals", L)
            site_group._v_attrs.y = new_y[-1]

        y = new_y
        sites = new_sites
        level += 1
    fh.close()
    

def compute_eigs(A, B):
    P = PolynomialFeatures()
    S = P.fit_transform(np.concatenate((A, B), axis=1))
    cov = (1/float(S.shape[0]))*S.T.dot(S)
    L, V = np.linalg.eigh(cov)

    power = np.cumsum(L)
    power /= power[-1]

    where = power >= EIG_THRESH
    if L.size - MAX_EIGS > 0:
        where[:(L.size - MAX_EIGS)] = False
    where[:3] = False
    V = V[:,where]
    W = S.dot(V)

    return L, V, W


if __name__=="__main__":
    main()
