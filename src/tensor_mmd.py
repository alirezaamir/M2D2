import os
import sys
import tqdm
import tables
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import linalg

SF = 256

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def main():
    dirname = "../temp/tensors"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    h5_file_list = ["../temp/tcgrn/" + x for x in os.listdir("../temp/tcgrn")]
    for fname in h5_file_list:
        h5_file = tables.open_file(fname)
        level = 1
        level_node = h5_file.get_node("/level{}".format(level))
        key = lambda x: int(x.replace("site",""))
        node_names = sorted(level_node._v_children, key=key)
      
        y = []
        subspaces = []
        for n in node_names:
            site_node = level_node[n]
            subspaces.append(site_node["eigvects"].read())
            y.append(site_node._v_attrs.y)
            
        h5_file.close()

        where = np.where(y)[0]
        start = np.min(where)
        stop = np.max(where)

        sr = 512*np.power(2, level)
        sr_seconds = sr / float(SF)
        obs_buff = int(np.ceil(300 / float(sr_seconds)))

        y = np.array(y[(start-obs_buff):(stop+obs_buff)])
        subspaces = subspaces[(start-obs_buff):(stop+obs_buff)]
        K = kernel(subspaces)
        mmd = []
        for N in range(1,len(subspaces)-1):
            M = K.shape[0] - N
            Kxx = K[:N,:N].sum()
            Kxy = 2*K[:N,N:].sum()
            Kyy = K[N:,N:].sum()
            mmd.append((((1/float(N*N))*Kxx) + 
                ((1/float(M*M))*Kyy) -
                ((2/float(N*M))*Kxy)
            ))
        
        ws = []
        N = len(subspaces)
        mmd = np.array(mmd)
        mmd_corr = np.zeros(mmd.size)
        for ix in range(1,mmd_corr.size):
            w = ((N-1) / float(ix*(N-ix)))
            ws.append(w)
            mmd_corr[ix] = mmd[ix] - w*mmd.max()

        mmd = mmd[200:-200]
        mmd_corr = mmd_corr[200:-200]
        
        plt.close()
        plt.plot(mmd, label="MMD")
        plt.plot(mmd_corr, label="MMD (Corrected)")
        plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
        plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")
        
        stub = fname.split("/")[-1].replace(".h5", "")
        plt.savefig("{}/{}_mmd.png".format(dirname, stub))


def kernel(X):
    N = len(X)
    K = np.ones((N,N))
    for ixi in tqdm.tqdm(range(N)):
        P = X[ixi]
        for ixj in range(ixi+1,N):
            Q = X[ixj]
            v = min(P.shape[1], Q.shape[1])
            K[ixi,ixj] = (1/float(v))*np.square(np.linalg.norm(P.T.dot(Q)))

    ixs = np.tril_indices(N)
    K[ixs] = K.T[ixs]
    return K
  

def plot_seizures():
    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            if len(node.attrs.seizures) > 0:
                data = node.read()
                seizures = node.attrs.seizures
                X, y = data[:,:-1], data[:,-1]
                for s in seizures:
                    ix = np.min(np.where(y == s)[0])
                    plt.close()
                    plt.plot(X[y==s,0], label="Channel 0")
                    plt.plot(X[y==s,1], label="Channel 1")
                    plt.legend()
                    plt.savefig("../temp/{}_s{}.png".format(node._v_name, s))





if __name__=="__main__":
    main()
