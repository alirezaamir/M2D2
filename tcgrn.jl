using ITensors
using Statistics
using MLDatasets
using ScikitLearn
using LinearAlgebra

@sk_import linear_model: LogisticRegression;

const EPS = 1e-3;
const MIN_DIM = 1;

function main()
    @info "Loading training data..."
    x_train, y_train = MNIST.traindata();
    X = convert(Array{Float64}, MNIST.convert2features(x_train)');
    N,K = size( X );

    Φ_dim = 2;
    Φ = ITensor[];
    ix_obs = Index(N, "obs");
    for k in 1:K
        global Φ, Φ_dim, ix_obs;
        ix_site = Index(Φ_dim, "site");
        T = fill!(ITensor(ix_obs, ix_site), 1.0);
        for n in 1:N
            T[ix_obs(n), ix_site(2)] = X[n,k];
        end
        push!(Φ, T);
    end

    tensors = Array{ITensor,1}[];
    features = Array{ITensor,1}[];

    layer = 1;
    while length(Φ) > 1
        @info "Coarse Graining Layer $layer";
        T, Φ = coarse_grain_layer( Φ );
        push!(tensors, T);
        push!(features, Φ);
        layer += 1;
    end

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

    @info "Generating Φ ..."
    Φ = [[ones(N,1) X[:,ix]] for ix in 1:K];
    X = coarse_grain_data( Φ, tensors )[1];

    y_pred = predict(clf, X);
    acc_test = mean(y_pred .== y_test);
    @info "Test accuracy: $acc_test"
end


function coarse_grain_layer( Φ )
    num_sites_new = Int(ceil(length(Φ) / 2.));

    N,K = size(Φ[1]);
    traces = [norm(x*x) for x in Φ];
    
    tensors = ITensor[];
    Φ_new = ITensor[];

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        trace = sum(traces[filter(x->!(x in [ix, next]), 1:length(traces))]);
        U, ϕ = coarse_grain_site( Φ[ix], Φ[next], trace);
        push!(tensors, U);
        push!(Φ_new, ϕ);
    end

    return (tensors, Φ_new)
end


function coarse_grain_site( A, B, trace )
    s1 = findindex(A, "site");
    s2 = findindex(B, "site");
    ix_obs = findindex(A, "obs");
    Ω = (A*prime(A, s1))*(B*prime(B, s2));
    trace = (trace == 0.0) ? 1 : trace;

    N = size(A,1);
    Ω *= (trace / Float64(N));
    Ω /= norm( Ω );  # this should help with numerical precision...
    U, λ, V = svd( Ω, s1, s2, cutoff=EPS );

    # This isn't very efficient... Should optimize somehow at some point
    ϕ = zip_apply(A, B, U);

    return (U, ϕ)
end


function zip_apply(A::ITensor, B::ITensor, U::ITensor)
    s1 = findindex(A, "site");
    s2 = findindex(B, "site");
    ix_obs = findindex(A, "obs");
    ix_u = findindex(U, "Link,u");
    ix_site = replacetags(ix_u, "Link,u", "site");

    N = dim(ix_obs)
    K = dim(ix_site);
    ϕ = ITensor(ix_obs, ix_site);

    # This isn't very efficient... Should optimize somehow at some point
    for n in 1:N
        t1 = ITensor(s1);
        for k in 1:dim(s1)
            t1[s1(k)] = A[ix_obs(n), s1(k)];
        end
        t2 = ITensor(s2);
        for k in 1:dim(s2)
            t2[s2(k)] = B[ix_obs(n), s2(k)];
        end
        u = t1*U*t2
        for k in 1:K
            ϕ[ix_obs(n), ix_site(k)] = u[ix_u(k)];
        end 
    end
    return ϕ
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
