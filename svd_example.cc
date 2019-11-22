#include "itensor/all.h"
using namespace itensor;

int main() {
    auto s1 = Index(2);
    auto s2 = Index(2);

    auto A = ITensor(s1);
    auto B = ITensor(s2);

    A.set(s1=1, 1.0);
    A.set(s1=2, 2.0);
    B.set(s2=1, 3.0);
    B.set(s2=2, 4.0);

    auto AB = (A*B)*(prime(A)*prime(B));
    PrintData(AB);

    auto [U, S, V] = svd( AB, {s1, s2} );
    PrintData( U );
    PrintData( S );
    PrintData( V );

    return 0;
}