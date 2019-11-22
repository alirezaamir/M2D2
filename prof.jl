using LinearAlgebra

function main()
    A = rand(40, 10000);
    B = rand(40, 10000);

    accum_covmat(A, B);
    for ix in 1:5
        @time accum_covmat(A, B);
    end
end

function accum_covmat(A::Array{Float64,2}, B::Array{Float64,2})
    K, N = size( A );
    M = size( B, 1 );

    Ω = zeros(K,);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], Ω);
    end
end

main()
