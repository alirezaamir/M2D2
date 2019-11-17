using Statistics
using MLDatasets
using ScikitLearn
using LinearAlgebra

@sk_import linear_model: LogisticRegression

EPS = 1e-3;
MIN_DIM = 1;

function main()
    @info "Loading training data..."
    x_train, y_train = MNIST.traindata();
    X = convert(Array{Float64}, MNIST.convert2features(x_train)');
    N,K = size( X );

    tensors = Array{Array{Float64,2}}[];
    features = Array{Array{Float64,2}}[];
    Φ = [[ones(N,1) X[:,ix]] for ix in 1:K];

    layer = 1;
    while length(Φ) > 1
        @info "Coarse Graining Layer $layer";
        T, Φ = coarse_grain_layer( Φ );
        push!(tensors, T);
        push!(features, Φ);
        layer += 1;
    end

    clf_base = LogisticRegression(fit_intercept=true);
    fit!(clf_base, X, y_train)
    y_pred = predict(clf_base, X)
    acc_train_base = mean(y_pred .== y_train);
    @info "Baseline train accuracy: $acc_train_base"

    X = features[end][1];

    @info "Coarse grained dimension: $(size(X,2))"
    clf = LogisticRegression(fit_intercept=true);
    fit!(clf, X, y_train);

    y_pred = predict(clf, X);
    acc_train = mean(y_pred .== y_train);
    @info "Train accuracy: $acc_train"

    @info "Loading test data..."
    x_test, y_test = MNIST.testdata();
    X = convert(Array{Float64}, MNIST.convert2features(x_test)');
    N, K = size( X );

    y_pred = predict(clf_base, X);
    acc_test_base = mean(y_pred .== y_test);
    @info "Baseline test accyuracy: $acc_test_base"

    @info "Generating Φ ..."
    Φ = [[ones(N,1) X[:,ix]] for ix in 1:K];
    X = coarse_grain_data( Φ, tensors )[1];

    y_pred = predict(clf, X);
    acc_test = mean(y_pred .== y_test);
    @info "Test accuracy: $acc_test"
end


function coarse_grain_layer( Φ )
    num_sites_new = Int(ceil(length(Φ) / 2.));
    tensors = Array{Float64,2}[];
    features = Array{Float64,2}[];

    N,K = size(Φ[1]);
    traces = zeros(length(Φ));
    for ix in 1:length(Φ)
        Q = Φ[ix];
        traces[ix] = sum([Q[n,:]'*Q[n,:] for n in 1:N]);
    end

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        Q = [Φ[ix] Φ[next]];

        trace = sum(traces[filter(x->!(x in [ix, next]), 1:length(traces))]);
        T, P = coarse_grain_site( Q, trace );
        push!(tensors, T);
        push!(features, P);
    end
    return (tensors, features)
end


function coarse_grain_site( A, B, trace )
    N, K = size( A );
    Ω = zeros(K, K);

    for n in 1:N
        ω₁ = A[n,:]*A[n,:]';
        ω₂ = B[n,:]*B[n,:]';
        Ω += kron(ω₁, ω₂);
    end

    trace = (trace == 0.0) ? 1 : trace;
    Ω *= (trace / Float64(N));
    Ω /= norm( Ω )
    λ, U = eigen( Ω );
    
    if any(x -> x < -1, λ)
        @error "Covariance matrix is not PSD! Eigenvalues: $(λ)"
        throw(TypeError("Covariance matrix is not PSD"))
    end

    ix = findall(x -> x > EPS, cumsum( λ ) / tr( Ω ))[1];
    U = U[:,ix:end];
    P = Q*U;
    return (U, P)
end


function coarse_grain_data( Φ, tensors )
    for layer in tensors
        tix = 1
        Φ_new = Array{Float64,2}[];
        for ix in 1:2:length(Φ)
            next = ix == length(Φ) ? ix-1 : ix + 1;            
            Q = [Φ[ix] Φ[next]];
            push!(Φ_new, Q*layer[tix]);
            tix += 1;
        end
        Φ = Φ_new;
    end
    return Φ
end

main();
