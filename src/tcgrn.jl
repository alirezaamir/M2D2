using DSP
using HDF5
using Dates
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
            do_subspace_detection(subject_id, ε, dirname);
            # features, labels = coarse_grain_data(subject_id, ε, dirname);
            # predict_kde(features, labels, subject_id, dirname);
        end
    end
end


function do_subspace_detection(subject_id, eps, dirname)
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5", "r");
    h5_node = h5_file[subject_id];
    Φ = prepare_data(h5_node);
    close(h5_file);

    X = [S.ϕ for S in Φ];
    y = [S.y for S in Φ];

    h5_file = HDF5.h5open("../temp/$eps/decomp_$subject_id.h5")
    num_layers = length(names(h5_file));

    for l in 1:num_layers
        @info "Predicting on layer: $l"
        next(ix) = ix == length(y) ? ix-1 : ix + 1;
        y = [y[ix] | y[next(ix)] for ix in 1:2:length(y)];
        
        layer_node = h5_file["layer$l"]
        eigs = [read(layer_node["site$s"]["U"]) for s in 1:length(layer_node)];
        V, dists = compute_average_subspace( eigs );
        time = collect(1:length(dists))
        labels = (Dates.Second(1)*2^l).*time;
        scatter(time[.!y], dists[.!y],
            color="green", label="Non-Ictal", alpha=0.1);
        scatter!(time[y], dists[y], color="cyan", label="Ictal");
        savefig("$dirname/layer$l/global_dists.png"); closeall();

        f(x) = compute_average_subspace( x )[2];
        dists = collect(Iterators.flatten(
            [f( eigs[ix-100:ix] ) for ix in 101:100:length(eigs)]));
        y = collect(Iterators.flatten(
            [y[ix-100:ix] for ix in 101:100:length(eigs)]));
        time = collect(Iterators.flatten(
            [time[ix-100:ix] for ix in 101:100:length(eigs)])); 
        scatter(time[.!y], dists[.!y],
            color="green", label="Non-Ictal", alpha=0.1);
        scatter!(time[y], dists[y], color="cyan", label="Ictal");
        savefig("$dirname/layer$l/local_dists.png"); closeall();        
    end
end


function compute_average_subspace(B::Array{Array{Float64,2},1})
    P_bar = (1/length(B))*sum([U*U' for U in B]);
    λ, F = eigen( P_bar );
    s = sum( λ .> 0.5 );
    perm = sortperm( -1*λ );
    F = F[:,perm];
    V = F[:,1:s];
    dists = [0.5*norm(U*U' - V*V') for U in B];
    return V, dists
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
            push!(features, X);
            push!(labels, y);
        end
    end

    return features, labels
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
