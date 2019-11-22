#include <omp.h>
#include <vector>
#include <math.h>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include <python2.7/Python.h>
#include <boost/log/trivial.hpp>

#include "itensor/all.h"

#define EPS 0.0001
#define REAL(x) static_cast<double>(x)
#define INT(x) static_cast<int>(x)
#define NUM_SITES_NEW(K) INT(ceil(REAL(K) / 2.));

using namespace itensor;
using namespace std;

// the outer vector represents observations
// the inner vector represents features 
typedef vector<vector<ITensor>> PHI_t;

struct Layer {
    PHI_t &phi;
    vector<ITensor> &tensors;

    Layer(PHI_t &phi, vector<ITensor> &tensors) : phi(phi), tensors(tensors) {};
};

void coarse_grain_site(size_t ix, size_t next, Real trace, PHI_t &phi, Layer &layer);
void coarse_grain_layer(PHI_t &phi, Layer &layer);
Real accum_trace(vector<Real> &traces, size_t ix, size_t next);

tuple<size_t,size_t> sizep(PHI_t &phi) { 
    return {phi.size(), phi.at(0).size()}; 
}

size_t sizep(PHI_t &phi, uint axis) {
    if (axis == 1) return phi.size();
    if (axis == 2) return phi.at(0).size();
    BOOST_LOG_TRIVIAL(error) << "Invalid axis: ";
    return -1;
}

static PyObject *tcgrn(PyObject *self, PyObject *args) {
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &data))
        return NULL;

    // check correct type
    if (!PyFloat_Check(PyList_GetItem(PyList_GetItem(data, 0), 0))) {
        auto message = "Data should be floats in 0-1";
        PyErr_SetString(PyExc_TypeError, message);
        return NULL;
    }

    const uint phi_dim = 2;
    const size_t N = PyList_Size(data);
    const size_t K = PyList_Size(PyList_GetItem(data, 0));

    // copy the data into ITensors
    vector<Index> site_indexes;
    for (size_t kk=0; kk<K; kk++) {
        auto s = Index(phi_dim, "Site");
        site_indexes.push_back(s);
    }

    PHI_t phi0;
    BOOST_LOG_TRIVIAL(info) << "Copying data...";
    for (size_t ii=0; ii<N; ii++) {
        auto row = vector<ITensor>();
        auto img = PyList_GetItem(data, ii);
        for (size_t jj=0; jj<K; jj++) {
            auto s = site_indexes.at(jj);
            auto T = ITensor(s).fill(1.0);
            Real val = PyFloat_AsDouble(PyList_GetItem(img, jj));
            T.set(s=1, val);
            row.push_back(T);
        }
        phi0.push_back(row);
    }

    auto phi = phi0;
    uint layer = 1;
    PHI_t tensors;
    while (phi[0].size() > 1) {
        BOOST_LOG_TRIVIAL(info) << "Coarse graining layer: " + to_string(layer++);

        // first allocate the storage for results
        auto T = vector<ITensor>();
        PHI_t phi_new;
        for (size_t n=0; n<N; n++) {
            phi_new.push_back(vector<ITensor>());
        }

        Layer L(phi_new, T);
        coarse_grain_layer(phi, L);

        phi = phi_new;
        tensors.push_back(L.tensors);
    }

    // convert the results to Python list
    PyObject *cgrn_data = PyList_New(N);
    if (cgrn_data == NULL) {
        PyErr_SetString(PyExc_Exception, "Could not allocate new list");
        return NULL;
    }

    auto site_ix = findIndex(phi[0][0], "Site");
    Print(site_ix);
    BOOST_LOG_TRIVIAL(info) << "Final site dim: " << dim(site_ix);

    for (size_t n=0; n<N; n++) {
        PyObject *row = PyList_New(dim(site_ix));
        if (row == NULL) {
            PyErr_SetString(PyExc_Exception, "Could not allocate row");
        }
        for (size_t k=1; k<=dim(site_ix); k++) {
            Real val = phi[n].back().elt(site_ix=k);

            if (PyList_SetItem(row, k-1, PyFloat_FromDouble(val)) < 0) {
                BOOST_LOG_TRIVIAL(error) << "Invalid value for: " << k << " => " << val;
                PrintData(phi[n].back());
                PyErr_SetString(PyExc_Exception, "Could not set coordinate");
                return NULL;
            }
        }
        if (PyList_SetItem(cgrn_data, n, row) < 0) {
            PyErr_SetString(PyExc_Exception, "Could not set row");
            return NULL;
        }
    }
    
    return cgrn_data;
}

void coarse_grain_layer(PHI_t &phi, Layer &layer) {
    auto [N, K] = sizep( phi );
    
    vector<Real> traces(K, 0.0);

    #pragma omp parallel for
    for (size_t ii=0; ii<N; ii++) {
        for (size_t jj=0; jj<K; jj++) {
            traces[jj] += norm(phi[ii][jj]);
        }
    }

    for (size_t jj=0; jj<K; jj+=2) {
        size_t next = jj == K-1 ? jj-1 : jj+1;
        BOOST_LOG_TRIVIAL(info) << "Coarse graining sites: " << jj << " -> " << next;
        auto trace = accum_trace(traces, jj, next);
        coarse_grain_site(jj, next, trace, phi, layer);
    }
}

Real accum_trace(vector<Real> &traces, size_t ix, size_t next) {
    Real trace = 0.0;
    for (size_t jj=0; jj<traces.size(); jj++) {
        if (jj != ix && jj != next) trace += traces.at(jj);
    }
    return trace;
}

void coarse_grain_site(size_t ix, size_t next, 
    Real trace, PHI_t &phi, Layer &layer) {
    auto s1 = findIndex(phi[0][ix], "Site");
    auto s2 = findIndex(phi[0][next], "Site");
    auto C = ITensor(s1, s2, prime(s1), prime(s2));

    trace = (trace == 0.0) ? 1.0 : trace;
    auto N = sizep( phi, 1 );
    
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (uint n=0; n<N; n++){
        auto S1 = phi[n].at(ix)*prime(phi[n].at(ix));
        auto S2 = phi[n].at(next)*prime(phi[n].at(next));
        
        #pragma omp critical
        C += S1*S2;
    }
    auto stop = chrono::high_resolution_clock::now();
    auto elapsed = stop - start;
    BOOST_LOG_TRIVIAL(info) << "Computing covamt took: " << elapsed.count();

    C *= REAL(N)*trace;
    C /= norm(C);

    start = chrono::high_resolution_clock::now();
    auto [U,S,V] = svd(C, {s1, s2}, {"Cutoff", EPS});
    stop = chrono::high_resolution_clock::now();
    elapsed = stop - start;
    BOOST_LOG_TRIVIAL(info) << "Compuring SVD took: " << elapsed.count();

    auto link = findIndex(U, "U,Link");
    auto six = replaceTags(link, "U,Link", "Site");
    U = U*delta(link, six);
    Print(U);

    for (size_t n=0; n<N; n++) {
        layer.phi[n].push_back(phi[n][ix]*U*phi[n][next]);
    }
    layer.tensors.push_back(U);
}

static PyMethodDef tcgrn_methods[] = {
    {"tcgrn", tcgrn, METH_VARARGS, "Python Interface for TCGRN"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inittcgrn( void ) {
    (void) Py_InitModule("tcgrn", tcgrn_methods);
}
