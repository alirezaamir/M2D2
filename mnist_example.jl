using NPZ
using Images
using Profile
using TestImages
using Statistics
using MLDatasets
using ScikitLearn
using LinearAlgebra

@sk_import linear_model: LogisticRegression

@enum Mode direct_sum tensor_prod

function main()
    @info "Loading training data..."
    x_train, y_train = MNIST.traindata();
    X = convert(Array{Float64}, MNIST.convert2features(x_train));
    K,N = size( X );

    tensors = Array{Array{Float64,2}}[];
    decomp_mode = Array{Mode,1}[];
    Φ = [[ones(1,N); X[ix,:]'] for ix in 1:K];

    eps = 1e-3;
    layer = 1;
    while length(Φ) > 1
        global layer, Φ, tensors, eps;
        @info "Coarse Graining Layer $layer => ε = $(eps)";
        @time T, Φ, M = coarse_grain_layer( Φ, eps );
        push!(tensors, T);
        push!(decomp_mode, M);
        layer += 1;
    end
    
    @info "Final dim: $(size( Φ[1] ))"

    clf = LogisticRegression();
    fit!(clf, Φ[1]', y_train);
    acc_train = score(clf, Φ[1]', y_train);

    x_test, y_test = MNIST.testdata();
    X = convert(Array{Float64}, MNIST.convert2features(x_test));
    K,N = size( X );

    Φ_test = [[ones(1,N); X[ix,:]'] for ix in 1:K];
    Φ_test = coarse_grain_data(Φ_test, tensors, decomp_mode);

    acc_test = score(clf, Φ_test[1]', y_test);
    @info "Train Accuracy: $acc_train"
    @info "Test Accuracy: $acc_test"
end


function coarse_grain_layer( Φ, eps )
    tensors = Array{Float64,2}[];
    features = Array{Float64,2}[];
    decomp_mode = Mode[];

    K,N = size(Φ[1]);

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        T, P, m = coarse_grain_site( Φ[ix], Φ[next], eps );
        push!(tensors, T);
        push!(features, P);
        push!(decomp_mode, m);
    end
    return (tensors, features, decomp_mode)
end


function coarse_grain_site( A::Array{Float64,2}, B::Array{Float64,2}, eps::Float64 )
    K, N = size( A );
    M = size( B, 1 );

    mode = K*M > 10000 ? direct_sum : tensor_prod
    @info "Array size: $K x $M => Mode: $mode"

    Ω = mode == direct_sum ? accum_covmat_sum(A, B) : accum_covmat_prod(A, B);
    if sum(Ω) == 0
        @error "Covariance matrix is all zeros!"
        throw(TypeError("Invalid covariance matrix"))
    end

    Ω /= norm(Ω);
    λ, U = eigen( Symmetric(Ω) );
    π = sortperm(λ);
    U = U[:,π];
    λ = λ[π];
    
    if any(x -> x < -1e-5, λ)
        @error "Covariance matrix is not PSD! Eigenvalues: $(λ)"
        throw(TypeError("Covariance matrix is not PSD"))
    end

    ix = findall(x -> x > eps, cumsum( λ ) ./ sum( λ ))[1];
    U = U[:,ix:end];
    P = mode == direct_sum ? project_sum(U, A, B) : project_prod(U, A, B);

    return (U, P, mode)
end


function coarse_grain_data( Φ, tensors, decomp_mode )
    for ix_l in 1:length(tensors)
        ix_u = 1
        Φ¹ = Array{Float64,2}[];
        for ix in 1:2:length(Φ)
            next = ix == length(Φ) ? ix-1 : ix + 1;
            A = Φ[ix];
            B = Φ[next];
            U = tensors[ix_l][ix_u];
            mode = decomp_mode[ix_l][ix_u]; 
            ϕ = mode == direct_sum ? project_sum(U, A, B) : project_prod(U, A, B);
            push!(Φ¹, ϕ);
            ix_u += 1;
        end
        Φ = Φ¹;
    end
    return Φ
end


function accum_covmat_prod(A::Array{Float64,2}, B::Array{Float64,2})
    K,N = size( A );
    M = size( B, 1 );

    ω = zeros(K, M);
    Ω = zeros(K*M, K*M);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], ω);
        BLAS.ger!(1.0, vec(ω), vec(ω), Ω);
        fill!(ω, 0.0);
    end

    return Ω
end


function project_prod(U::Array{Float64,2}, A::Array{Float64,2}, B::Array{Float64,2})
    K, N = size( A );
    M = size( B, 1 );

    P = zeros(size(U,2), N);
    ω = zeros(K, M);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], ω);
        BLAS.gemv!('T', 1.0, U, vec(ω), 0.0, @view P[:,n]);
        fill!(ω, 0.0);
    end
    return P
end


function accum_covmat_sum(A::Array{Float64,2}, B::Array{Float64,2})
    K,N = size( A );
    M = size( B, 1 );

    Ω = zeros(K+M, K+M);
    for n in 1:N
        ω = [A[:,n]; B[:,n]];
        BLAS.ger!(1.0, ω, ω, Ω);
    end

    return Ω
end

function project_sum(U::Array{Float64,2}, A::Array{Float64,2}, B::Array{Float64,2})
    K, N = size( A );
    M = size( B, 1 );

    P = zeros(size(U,2), N);
    for n in 1:N
        ω = [A[:,n]; B[:,n]];
        BLAS.gemv!('T', 1.0, U, ω, 0.0, @view P[:,n]);
    end
    return P
end

main();
