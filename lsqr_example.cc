#include "itensor/all.h"
using namespace itensor;

#define N_OBS 10
#define N_VAR 2
#define TOL 0.001

int main() {
    auto i = Index(N_OBS, "index i");
    auto j = Index(N_VAR, "index j");
    
    // Generate some synthetic outcome data
    auto A = randomITensor(i, j);
    auto w = randomITensor(j);
    auto y = A*w;

    auto [U,S,V] = svd(A, {i});
    Print(U);
    Print(V);
    Print(S*V);

    std::cout << "True weights " << std::endl;
    PrintData(w);

    Index ixi = Index(2, "ixi");
    Index ixj = Index(2, "ixj");
    auto p = ITensor(ixi); p.set(1, 1); p.set(2, 2);
    auto q = ITensor(ixj); q.set(1, 10); q.set(2, 9);
    PrintData(p*q);

    // some variable defs
    ITensor predicted, eps, grad;
    double alpha = 0.01;
    double cost = 1000.0;

    auto b = randomITensor(j);
    int iter = 0;
    while (cost > TOL) {
        predicted = A*b;
        eps = y - predicted;
        grad = A*eps;
        b = b + alpha*grad;
        cost = norm(eps);
        ++iter;
        if (iter % 100 == 0) 
            std::cout << "Iteration: " << iter << " Cost: " << norm(eps) << std::endl;
    }

    std::cout << "Fitted Weights: " << std::endl;
    PrintData(b);

    return 0;
}