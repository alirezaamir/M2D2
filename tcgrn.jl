using Images
using Profile
using ITensors
using TestImages
using Statistics
using MLDatasets
using Distributed
using ScikitLearn
using LinearAlgebra

@sk_import linear_model: LogisticRegression

addprocs(Int(length(Sys.cpu_info()) / 2))
@everywhere push!(LOAD_PATH, pwd());
@everywhere using TCGRNUtils

function main()
    @info "Loading training data..."
    x_train, y_train = CIFAR10.traindata();
    X = convert(Array{Float64}, Gray.(CIFAR10.convert2image(x_train)));
    P,Q,N = size( X );
    X = reshape(X, P*Q, N);
    K = P*Q;

    obs_index = Index(N, "obs");
    tensors = ITensor[];
    Φ = ITensor[];
    for k in 1:K
        site_index = Index(2, "site");
        ϕ = fill!(ITensor(site_index, obs_index), 1.0);
        ϕ.store.data[2:2:2*N] = X[k,:];
        push!(Φ, ϕ);
    end

    layer = 1;
    while length(Φ) > 1
        # global layer, Φ, tensors;
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


function coarse_grain_layer( Φ )
    num_sites_new = Int(ceil(length(Φ)/2.0));
    tensors = Array{Any,1}(nothing, num_sites_new);
    features = Array{Any,1}(nothing, num_sites_new);

    K,N = size(Φ[1]);
    next(x) = x == length(Φ) ? x-1 : x + 1;
    res = pmap(x->TCGRNUtils.coarse_grain_site(Φ[x], Φ[next], x), 1:2:4)
    return (tensors, features)
end


main();
