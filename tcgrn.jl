using Images
using Profile
using TestImages
using Statistics
using MLDatasets
using ScikitLearn
using LinearAlgebra

@sk_import linear_model: LogisticRegression

const EPS = 1e-3;


function main()
    @info "Loading training data..."
    x_train, y_train = CIFAR10.traindata();
    X = convert(Array{Float64}, Gray.(CIFAR10.convert2image(x_train)));
    P,Q,N = size( X );
    X = reshape(X, P*Q, N);
    K = P*Q;

    tensors = Array{Array{Float64,2}}[];
    Φ = [[ones(1,N); X[ix,:]'] for ix in 1:K];

    layer = 1;
    while length(Φ) > 1
        global layer, Φ, tensors;
        @info "Coarse Graining Layer $layer";
        @time T, Φ = coarse_grain_layer( Φ );
        push!(tensors);
        layer += 1;
    end

    @info "Final size: $(size(Φ))"
    X = Φ[1]';
    clf = LogisticRegression();
    clf.fit(X, y_train);

    y_pred = clf.predict(X);
    acc = mean(y_pred .== y_train);
    @info "Accuracy: $acc"
end

function rescale!(X)
    for ix in 1:size(X,1)
        M = @view X[ix,:]
        if (M[1] == mean(M)) continue; end
        X[ix,:] = (M .- minimum(M)) ./ (minimum(M) - maximum(M)); 
    end
end


function coarse_grain_layer( Φ )
    tensors = Array{Float64,2}[];
    features = Array{Float64,2}[];

    K,N = size(Φ[1]);

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        T, P = coarse_grain_site( Φ[ix], Φ[next] );
        push!(tensors, T);
        push!(features, P);
    end
    return (tensors, features)
end


function coarse_grain_site( A, B )
    K, N = size( A );
    M = size( B, 1 );

    Q = [A; B];
    Ω = Q*Q';
    if sum(Ω) == 0
        @error "Covariance matrix is all zeros!"
        throw(TypeError("Invalid covariance matrix"))
    end

    Ω /= Float64(N);
    λ, U = eigen( Ω );
    
    if any(x -> x < -1e-5, λ)
        @error "Covariance matrix is not PSD! Eigenvalues: $(λ)"
        throw(TypeError("Covariance matrix is not PSD"))
    end

    ix = findall(x -> x > EPS, cumsum( λ ) ./ sum( λ ))[1];
    U = U[:,ix:end];
    P = U'*Q;    
    return (U, P)
end


function accum_covmat(A::Array{Float64,2}, B::Array{Float64,2})
    K,N = size( A );
    M = size( B, 1 );

    ω = zeros(K, M);
    Ω = zeros(K*M, K*M);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], ω);
        BLAS.ger!(1.0, ω[:], ω[:], Ω);
        fill!(ω, 0.0);
    end

    return Ω
end


# main();
