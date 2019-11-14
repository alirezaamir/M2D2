#include <omp.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <python2.7/Python.h>
#include <boost/log/trivial.hpp>

#include "itensor/all.h"

#define TOL 0.0001
#define MAX_SWEEPS 1000
#define __DEBUG__ false
#define STEP_SIZE 0.01
#define MAX_INNER_ITER 100
#define PI2 1.5707963267948966
#define BOND_DIM 1

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

string gen_label(uint i) { return "a" + to_string(i) + "->" + to_string(i+1); }

tuple<Real, vector<Real>> compute_cost(PyObject *labels,
                                       vector<ITensor> mps_tensors, 
                                       vector<vector<ITensor>> data_tensors) {
    Real cost = 0;
    int ix = 0;
    vector<Real> fitted_values;
    for (auto row : data_tensors) {
        Real lbl = PyFloat_AsDouble(PyList_GetItem(labels, ix)); ix++;
        auto fx = row.at(0)*mps_tensors.at(0);
        for (unsigned jj=1; jj<row.size(); jj++) {
            fx *= row.at(jj)*mps_tensors.at(jj);
        }
        if (fx.order() > 0) {
            Print(fx);
            return {-9999, fitted_values};
        }
        Real pred = fx.elt();
        cost += pow(lbl - pred, 2);
        fitted_values.push_back(pred);
    }
    cost /= (double) data_tensors.size();
    return {cost, fitted_values};
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
        PyErr_SetString(PyExc_TypeError, "Labels must be floats!");

    uint N = PyList_Size(data);
    uint K = PyList_Size(PyList_GetItem(data, 0));

    // seed the RNG
    seedRNG(13298);

    vector<Index> site_indexes;
    vector<ITensor> mps_tensors;
    Index idx_left = Index(BOND_DIM, "a0->1");
    Index site = Index(2, "s0");
    mps_tensors.push_back(randomITensor(idx_left, site));
    site_indexes.push_back(site);
    for (uint six=1; six<K-1; six++) {
        site = Index(2, "s" + to_string(six));
        site_indexes.push_back(site);
        Index idx_right = Index(BOND_DIM, gen_label(six));
        auto T = randomITensor(site, idx_left, idx_right);
        PrintData(T);
        mps_tensors.push_back(T);
        idx_left = idx_right;
    }

    site = Index(2, "s" + to_string(K-1));
    mps_tensors.push_back(randomITensor(idx_left, site));
    site_indexes.push_back(site);

    BOOST_LOG_TRIVIAL(info) << "Data points: " << N;
    BOOST_LOG_TRIVIAL(info) << "Input dimension: " << K;
    BOOST_LOG_TRIVIAL(info) << "MPS Sites: " << mps_tensors.size();

    if (site_indexes.size() != mps_tensors.size()) {
        PyErr_SetString(
            PyExc_Exception, "Number of site indexes not equal to mps states");
        return NULL;
    }

    BOOST_LOG_TRIVIAL(info) << "Copying data..."; 
    vector<vector<ITensor>> data_tensors;
    for (uint ix=0; ix<N; ix++) {
        auto row = vector<ITensor>(); 
        auto img = PyList_GetItem(data, ix);

        for (uint ixj=0; ixj<K; ixj++) {
            auto s = site_indexes.at(ixj);
            auto D = ITensor(s);

            Real val = PyFloat_AsDouble(PyList_GetItem(img, ixj));
            D.set(s=1, cos(PI2*val));
            D.set(s=2, sin(PI2*val));
            row.push_back(D);
        }
        data_tensors.push_back(row);
        if (ix == 0) {
            for (auto v : row) PrintData(v);
        }
    }

    // copy the labels over to native C arrays for speed
    vector<Real> label_array;
    for (uint ix=0; ix<N; ix++)
        label_array.push_back(PyFloat_AsDouble(PyList_GetItem(labels, ix)));

    // run the optimization algorithm
    Real site_cost = 0.;
    auto [site_cost_new, fitted_vals] = compute_cost(
        labels, mps_tensors, data_tensors);
    if (site_cost_new == -9999) {
        PyErr_SetString(PyExc_Exception, "Invalid condition on f(x)");
        return NULL;
    }

    UpDownSequence seq = UpDownSequence(0, K-1);
    BOOST_LOG_TRIVIAL(info) << "Beginning optimization...";
    BOOST_LOG_TRIVIAL(info) << "Inital Cost: " << site_cost_new;

    Real *cost_arr = new Real[N];
    ITensor *grads_arr = new ITensor[N];

    while (abs(site_cost - site_cost_new) > TOL && seq.sweeps < MAX_SWEEPS) {
        auto idx = seq.next();
        site_cost = site_cost_new;
        BOOST_LOG_TRIVIAL(info) << "Optimizing sites: " << idx.i << " -> " << idx.j;

        // form the initial bond tensor
        auto T1 = mps_tensors.at(idx.i);
        auto T2 = mps_tensors.at(idx.j);
        auto B = T1*T2;

        uint iter = 0;
        Real cost = 0, cost_new = -9999;
        while (abs(cost - cost_new) > TOL && iter < MAX_INNER_ITER) {
            cost = cost_new;
            cost_new = 0;

            // #pragma omp parallel for
            for (uint ix=0; ix<N; ix++) {
                auto row = data_tensors.at(ix);
                auto phi = row.at(idx.i)*row.at(idx.j);
                for (uint jj=0; jj<min(idx.i, idx.j); jj++) {
                    phi *= mps_tensors.at(jj)*row.at(jj);
                }

                for (uint jj=K-1; jj>max(idx.i, idx.j); jj--) {
                    phi *= mps_tensors.at(jj)*row.at(jj);
                }

                auto fx = phi*B;
                Real pred = fx.elt();
                Real actual = label_array[ix];
                Real eps = actual - pred;
                cost_arr[ix] = pow(eps, 2);
                grads_arr[ix] = eps*phi;
            }

            // accumulate costs and gradients
            ITensor grads = ITensor(inds(B));
            for (uint ix=0; ix<N; ix++) {
                grads += grads_arr[ix];
                cost_new += cost_arr[ix];
            }

            cost_new /= static_cast<Real>(N);
            B += (1./((double) N))*STEP_SIZE*grads;
            ++iter;
        }

        BOOST_LOG_TRIVIAL(info) << "Inner loop converged in " <<
            iter << " iterations: " << cost_new;

        // decompose the updated bond tensor back into the two site tensors
        auto [U,S,V] = svd(B, inds(T1));
        T1 = U;
        T2 = S*V;
        mps_tensors.at(idx.i) = T1;
        mps_tensors.at(idx.j) = T2;

        auto [cc, fv] = compute_cost(labels, mps_tensors, data_tensors);
        if (cc == -9999) {
            PyErr_SetString(PyExc_Exception, "Invalid condition on f(x)");
            return NULL;
        }

        site_cost_new = cc;
        fitted_vals = fv;
        BOOST_LOG_TRIVIAL(info) << "Converged! Cost: " << site_cost_new;
    }

    delete[] grads_arr;
    delete[] cost_arr;

    BOOST_LOG_TRIVIAL(info) << "Converged! Final cost: " << site_cost_new;
    for (auto T : mps_tensors) PrintData(T);

    PyObject *predictions = PyList_New(fitted_vals.size());
    for (uint ix=0; ix<fitted_vals.size(); ix++)
        PyList_SetItem(predictions, ix, PyFloat_FromDouble(fitted_vals.at(ix)));

    return predictions;
}

static PyMethodDef tnml_methods[] = {
    {"tnml", tnml, METH_VARARGS, "Python Interface for TNML"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inittnml( void ) {
    (void) Py_InitModule("tnml", tnml_methods);
}
