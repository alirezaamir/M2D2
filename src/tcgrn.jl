using DSP
using HDF5
using Plots
using Random
using Measures
using MLKernels
using StatsBase
using DataFrames
using Statistics
using Clustering
using Distributed
using KernelDensity
using GaussianMixtures


# nproc = min(10, Int(length(Sys.cpu_info()) / 8));
# addprocs(nproc, enable_threaded_blas=true);

@everywhere using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN


function main()
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5")
    subject_ids = HDF5.names(h5_file);
    close(h5_file)

    subject_ids = ["chb01"];
    for subject_id in subject_ids
        ε_list = [1e-5];
        for ε in ε_list
            # TCGRN.process_subject(subject_id, ε);
            dirname = "../output/$subject_id/$ε";
            features, labels = coarse_grain_data(subject_id, ε, dirname);
            predict_kde(features, labels, subject_id, dirname);
        end
    end
end


function coarse_grain_data(subject_id, ε, dirname)
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5", "r");
    h5_node = h5_file[subject_id];
    Φ = prepare_data(h5_node);
    close(h5_file);

    X = [S.ϕ for S in Φ];
    y = [S.y for S in Φ];

    features = Array{Array{Float64,2},1}[];
    labels = Array{Float64,1}[];

    HDF5.h5open("../temp/$(ε)/decomp_$subject_id.h5") do h5_file
        num_layers = length(names(h5_file));
        for l in 1:num_layers
            @info "Coarse graining layer: $l"
            layer_node = h5_file["layer$l"];
            X, y = predict_layer(X, y, layer_node);
            compare_projections( X, "$dirname/layer$l" );
            push!(features, X);
            push!(labels, y);
        end
    end

    return features, labels
end


function compare_projections( X, dirname )
    N = length( X );
    Q = zeros( 2, N );
    for ix in 4:1:(N-3)
        P = zeros( 6 );
        L = X[ix];
        vv = 1;
        for jj in ix-3:1:ix+3
            if jj != ix
                M = X[jj];
                P[vv] = 2*norm(L - M) / (norm( L ) + norm( M ));
            end
        end
        Q[1,ix] = sqrt( var( P ) );
        Q[2,ix] = mean( P );
    end
    Q = Q[:,4:N-3];
    plot(Q[2,:], ylims=(0,2), ribbon=1.96*Q[1,:], fillalpha=0.3, color="blue");
    savefig("$dirname/proj_similarity.png");
end


function compare_subspaces( eigenvectors::Array{Array{Float64,2},1}, 
                            eigenvals::Array{Array{Float64,1}}, dirname::String )
    N = length(eigenvectors);
    Q = zeros( 2, N );
    for ix in 4:1:(N-3)
        P = zeros( 6 );
        L = loadings( eigenvectors[ix], eigenvals[ix] );
        vv = 1;
        for jj in ix-3:1:ix+3
            if jj != ix
                M = loadings( eigenvectors[jj], eigenvals[jj] );
                λ = eigvals( L*(M'*M)*L' );
                P[vv] = sum( λ );
                vv += 1;
            end
        end
        P /= size( eigenvectors[1], 2 );
        Q[1,ix] = sqrt( var( P ) );
        Q[2,ix] = mean( P );
    end
    Q = Q[:,4:N-3];

    plot(Q[2,:], ylims=(0,2), ribbon=1.96*Q[1,:], fillalpha=0.3, color="blue");
    savefig("$dirname/subspace_similarity.png");
end


function loadings( U, λ )
    V = zeros( size( U ) );
    λ = sqrt.(λ[end-size(U,2)+1:end]);
    for ix in 1:size(U,2)
        V[:,ix] = U[:,ix]*λ[ix];
    end
    return V
end


function predict_layer(X, y, layer_node)
    y_new = Bool[];
    X_new = Array{Float64,2}[];
    ix = 1;

    for s in 1:length(layer_node)
        next = ix == length(X) ? ix-1 : ix + 1;
        site_node = layer_node["site$s"];
        mode = TCGRN.Mode(read(site_node["mode"]));
        U = read(site_node["U"]);
        A = X[ix]; B = X[next];
        try
            ϕ = mode == TCGRN.direct_sum ? 
                TCGRN.project_sum(U, A, B) : TCGRN.project_prod(U, A, B);
            push!(X_new, ϕ);
        catch exc
            @error join(["Error processing:" ,
                         "layer $(name(layer_node))",
                         "site $(name(site_node))",
                         "Mode: $mode - ix: $ix/$(length(X)) - next: $next",
                         "U: $(size(U)) - A: $(size(A)) - B: $(size(B))"], "\n");
            throw(exc);
        end
        push!(y_new, y[ix] | y[next]);
        ix += 2;
    end

    return X_new, y_new
end


function predict_kde(features, labels, subject_id, dirname)    
    for l in 1:5
        y = [Bool(x) for x in labels[l]];
        first_ix = findall(x -> x != 0, y)[1]-1;
        X = copy(reduce(hcat, features[l][1:first_ix])');
        
        K = size( X, 2 );
        KDE = [InterpKDE(kde_lscv(X[ix,:])) for ix in 1:K];

        window_ll = zeros(length(y));
        for (ixw, W) in enumerate(features[l])
            ll = zeros(size( W, 2 ));
            for ix in 1:K
                P = pdf(KDE[ix], W[ix,:]);
                P[P .< 1e-15] .= 0;
                ll += log.( P );
            end
            ll[ll .== -Inf] .= 0;
            window_ll[ixw] = mean( ll );
        end

        new_seizure, prev_seizure, next_seizure = seizure_times( y );
        seizure_ixs_actual = findall(x -> x > 0, new_seizure);
        thresh = quantile(window_ll[1:first_ix], 0.01);
        seizure_ixs_pred = findall(x -> x < thresh, window_ll);
        
        # for each actual seizure, how much time elapsed before we identify it?
        time_to_detect = zeros(length(seizure_ixs_actual));
        for (ii, six) in enumerate(seizure_ixs_actual)
            elapsed = seizure_ixs_pred .- six;
            time_to_detect[ii] = minimum(elapsed[elapsed .>= 0]);
        end

        # for each predicted seizure, how much time elapsed before one actually happened?
        time_to_seizure = zeros(length(seizure_ixs_pred));
        for (ii, six) in enumerate(seizure_ixs_pred)
            elapsed = seizure_ixs_actual .- six;
            if any(elapsed .> 0)
                time_to_seizure[ii] = minimum(elapsed[elapsed .> 0]);
            else
                time_to_seizure[ii] = -1;
            end
        end

        open("$dirname/layer$l/time_to_seizure.txt", "w") do fh
            for v in time_to_seizure println(fh, v); end
        end
        open("$dirname/layer$l/time_to_detect.txt", "w") do fh
            for v in time_to_detect println(fh, v); end
        end 

        units = DURATION_SEC*(2^l);
        time_to_detect .*= units;
        time_to_seizure .*= units;

        xpoints = (1:length(y))[.!y];
        scatter(xpoints, window_ll[.!y], color="green", label="Non-Ictal", alpha=0.1);

        xpoints = (1:length(y))[y];
        scatter!(xpoints, window_ll[y], color="cyan", markershape=:star5, alpha=0.4, 
            label="Ictal", ylabel="Log-Likelihood", xlabel="Time (Seconds)");
        hline!([thresh], color="red", linewidth=3, label="Threshold");
        savefig("$dirname/layer$l/log_like_history.png"); closeall();
    end
end


function seizure_times(y::Array{Bool,1})
    new_seizure = [y[ix] == 1 && y[ix-1] == 0 for ix in 1:length(y)];
    next_seizure = zeros(length(y));
    next_seizure[end] = Inf;
    for ix in length(y)-1:-1:1
        if new_seizure[ix] != 0
            next_seizure[ix] = ix;
        else
            next_seizure[ix] = next_seizure[ix+1]
        end
    end

    prev_seizure = zeros(length(y));
    prev_seizure[1] = Inf;
    for ix in 2:length(y)
        if new_seizure[ix] != 0
            prev_seizure[ix] = ix
        else
            prev_seizure[ix] = prev_seizure[ix-1];
        end
    end
    return new_seizure, prev_seizure, next_seizure
end
