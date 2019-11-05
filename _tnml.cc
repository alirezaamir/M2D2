#include <vector>
#include <math.h>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <python2.7/Python.h>

#include "itensor/all.h"

#define TOL 0.00000001
#define MAX_SWEEPS 1000
#define __DEBUG__ false
#define STEP_SIZE 0.01
#define PI2 1.5707963267948966
#define TENSOR_ONES(inds) ITensor(inds).fill(1.0)

using namespace std;
using namespace itensor;

struct Ind {unsigned i,j; Ind(unsigned i, unsigned j) : i(i), j(j) {};};

struct UpDownSequence {
    int low, high, direction, ptr, sweeps;

    UpDownSequence(int a, int b) : low(a), high(b) {
        ptr = low+1;
        direction = -1;
        sweeps = -1;
    }

    Ind next() {
        ptr += direction;
        if ((ptr == high) || (ptr == low)) {
            direction *= -1;
            sweeps++;
        }
        return Ind(ptr, ptr+direction);
    }
};

Real compute_cost(PyObject *labels,
                  vector<ITensor> mps_tensors, 
                  vector<vector<ITensor>> data_tensors) {
    Real cost = 0;
    int ix = 0;
    for (auto row : data_tensors) {
        Real lbl = PyFloat_AsDouble(PyList_GetItem(labels, ++ix));
        auto fx = row.at(0)*mps_tensors.at(0);
        for (unsigned jj=1; jj<row.size(); jj++) {
            fx *= row.at(jj)*mps_tensors.at(jj);
        }
        cost += pow(lbl - fx.elt(), 2);
    }
    cost /= (double) data_tensors.size();
    return cost;
}

static PyObject *tnml(PyObject *self, PyObject *args) {
    PyObject *data, *labels;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &data, 
                                        &PyList_Type, &labels)) {
        return NULL;
    }

    // check that types are correct
    if (!PyFloat_Check(PyList_GetItem(PyList_GetItem(data, 0), 0)))
        PyErr_SetString(PyExc_TypeError, "Data must be floats!");
    if (!PyFloat_Check(PyList_GetItem(labels, 0)))
        PyErr_SetString(PyExc_TypeError, "Labels must be floats");

    unsigned d = 2;
    unsigned bond_dim_init = 2;
    auto N = PyList_Size(data);
    auto K = PyList_Size(PyList_GetItem(data, 0));

    // step 1: Create the MPS representing the weights
    unsigned num_sites = K-1;
    vector<ITensor> mps_tensors;
    vector<Index> site_idxs;
    
    // fill the tensors in the MPS
    Index idx_left = Index(bond_dim_init, "idx0->1");
    Index idx_site = Index(d, "idx_s0");
    mps_tensors.emplace_back(randomITensor(idx_site, idx_left));
    site_idxs.emplace_back(idx_site);

    for (unsigned ix=1; ix<num_sites; ix++) {
        Index idx_right = Index(bond_dim_init, 
            "idx" + to_string(ix) + "->" + to_string(ix+1));
        idx_site = Index(d, "idx_s" + to_string(ix));
        site_idxs.emplace_back(idx_site);
        auto T = randomITensor(idx_site, idx_left, idx_right);
        T /= norm( T );
        mps_tensors.emplace_back(T);
        idx_left = idx_right;
        Print(T);
    }

    idx_site = Index(d, "idx_s" + to_string(num_sites));
    site_idxs.emplace_back(idx_site);
    mps_tensors.emplace_back(randomITensor(idx_left, idx_site));

    if (__DEBUG__) {
        unsigned ix = 0;
        for (auto T : mps_tensors) {
            printfln("Site: %d", ix++);
            Print(T);
        }
    }

    // step 2: Create the tensors representing the data
    vector<vector<ITensor>> data_tensors;

    // step 3: Copy the Python data to the Tensor (TODO: Optimize this)
    for (unsigned ii=0; ii<N; ii++) {
        data_tensors.emplace_back(vector<ITensor>());
        PyObject *img = PyList_GetItem(data, ii);
        for (unsigned jj=0; jj<=num_sites; jj++) {
            Index idx_site = site_idxs.at(jj);
            ITensor T = ITensor(idx_site);
            Real val = PyFloat_AsDouble(PyList_GetItem(img, jj));
            T.set(idx_site=1, cos(PI2*val));
            T.set(idx_site=2, sin(PI2*val));
            T /= norm(T);
            data_tensors.back().emplace_back(T);
        }
    }

    // step 3: Run the optimization algorithm
    Real site_cost = 0.0;
    Real site_cost_prev = INFINITY;
    UpDownSequence seq = UpDownSequence(0, num_sites);
    while (abs(site_cost - site_cost_prev) > TOL && seq.sweeps < MAX_SWEEPS) {
        auto idx = seq.next();
        site_cost_prev = site_cost;
        printfln("Optimizing site: %d->%d => %f", idx.i, idx.j, site_cost_prev);

        ITensor site = mps_tensors.at(idx.i);
        ITensor next_site = mps_tensors.at(idx.j);
        auto B = site*next_site;

        Real inner_cost_prev = INFINITY;
        Real inner_cost = 0;
        unsigned iter = 0;
        while (iter < 4) {
        // while (abs(inner_cost - inner_cost_prev) > TOL) {
            printfln("Iteration (begin): %d => %f", ++iter, inner_cost);
            inner_cost_prev = inner_cost;
            inner_cost = 0;
            B /= norm(B);
            
            // accumulate gradients
            ITensor grads = ITensor(inds(B));
            unsigned ix = 0;
            for (auto v : data_tensors) {
                auto phi = v.at(idx.i)*v.at(idx.j);

                // contract over the wings to obtain the reduced data set
                for (unsigned jj=0; jj<min(idx.i, idx.j); jj++) {
                    auto lw = mps_tensors.at(jj)*v.at(jj);
                    phi *= lw;
                }
                
                for (unsigned jj=num_sites; jj>max(idx.j, idx.i); jj--) {
                    auto rw = mps_tensors.at(jj)*v.at(jj);
                    phi *= rw;
                }

                auto fx = B*phi;
                if (fx.order() > 0) {
                    Print(fx);
                    PyErr_SetString(PyExc_Exception, "Invalid condition on fx");
                    return NULL;
                }

                Real lbl_pred = fx.elt();
                Real lbl = PyFloat_AsDouble(PyList_GetItem(labels, ++ix));
                Real eps = lbl - lbl_pred;

                inner_cost += pow(eps, 2);
                PrintData(phi);
                grads += STEP_SIZE*eps*phi;

                if (__DEBUG__ && ix > 4) break;
            }
            inner_cost /= (double) data_tensors.size();
            printfln("Iteration (end): %d => %f", iter, inner_cost);
            printfln("Cost change: %f -> %f", inner_cost_prev, inner_cost);
            PrintData(grads);
            if (__DEBUG__) Print(grads);
            B += (STEP_SIZE*grads);
            PrintData(B);
        }
        // printfln("Converged! Final cost: %f", inner_cost);

        // split the bond tensor back into the two site tensors

        auto [U,S,V] = svd(B, inds(mps_tensors.at(idx.i)));
        mps_tensors.at(idx.i) = U;
        mps_tensors.at(idx.j) = S*V;

        site_cost = compute_cost(labels, mps_tensors, data_tensors);
        break;
    }

    PrintData(mps_tensors.at(0)*mps_tensors.at(1));

    return PyLong_FromLong(0L);
}

static PyMethodDef tnml_methods[] = {
    {"tnml", tnml, METH_VARARGS, "Python Interface for TNML"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inittnml( void ) {
    (void) Py_InitModule("tnml", tnml_methods);
}
