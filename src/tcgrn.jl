using DSP
using HDF5
using Plots
using Random
using Measures
using MLKernels
using StatsBase
using Statistics
using Clustering
using Distributed
using KernelDensity
using GaussianMixtures
using MultivariateStats


# nproc = min(10, Int(length(Sys.cpu_info()) / 8));
# addprocs(nproc, enable_threaded_blas=true);

@everywhere using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN


function main()
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5")
    subject_ids = HDF5.names(h5_file);
    close(h5_file)

#    TCGRN.process_subject(subject_ids[1], 1e-7);
#    features, labels = coarse_grain_data(subject_ids[1], 1e-7);    
    subject_ids = ["chb09"];
    for subject_id in subject_ids
        ε_list = [1e-7];
        for ε in ε_list
            # TCGRN.process_subject(subject_id, ε);
            features, labels = coarse_grain_data(subject_id, ε);
            dirname = "../output/$subject_id/$ε";
            if !isdir(dirname)
                mkpath(dirname);
            end
            layer_analysis_gmm(features, labels, subject_id, dirname);
        end
    end
end


function plot_eigenvalues(subject_id, labels, dirname, ε)
    HDF5.h5open("../temp/$ε/decomp_$(subject_id).h5") do h5_file
        for l in 1:length(labels)
            layer_node = h5_file["layer$l"];
            eigs = Array{Float32,1}[];
            for site_node in layer_node
                push!(eigs, read(site_node, "eigvals"));  
            end

            maxv = maximum([length(x) for x in eigs]);
            num_eigs = zeros(length(eigs));
            λ = fill!(zeros(maxv, length(eigs)), NaN);
            for ix in 1:length(eigs)
                num_eigs[ix] = length(eigs[ix]);
                λ[1:length(eigs[ix]),ix] = reverse(
                    cumsum(eigs[ix]) ./ sum(eigs[ix]));
            end

            y = labels[l] .> 0.0;
            nanmean(x) = mean(filter(!isnan, x));
            μ_yes = mapslices(nanmean, λ[:,y], dims=2);
            μ_no  = mapslices(nanmean, λ[:,.!y], dims=2);

            which = [ix for ix in 1:length(μ_yes) if μ_yes[ix] > 1e-9];
            ticks = [10.0^-x for x in 9:-1:0];
            plot(which, log.(vec(μ_yes[which])), 
                    linewidth=3, color=:darkgreen,
                    yticks=(log.(ticks), ticks),
                    ylims=(log(10^-9),0), label="Seizure");
            plot!(which, log.(vec(μ_no[which])), 
                    linewidth=3, color=:darkblue,
                    yticks=(log.(ticks), ticks),
                    ylims=(log(10^-9),0), label="Non-Seizure");
            savefig("$dirname/eigenvals.png");
        end
    end
end


function coarse_grain_data(subject_id, ε)
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


function layer_analysis_gmm(features, labels, subject_id, dirname)
    stride = 3*SRATE;
    seg_length = DURATION_SEC*SRATE;
    time = Int64[];
    push!(time, 0);
    HDF5.h5open("../input/eeg_data_temples2.h5") do h5_file
        node = h5_file["$subject_id"];
        for grp in node
            t = Int.(
                round.(
                    collect(seg_length+1:seg_length:size(grp,2)) / SRATE)
                ) .+ time[end];
            for x in t push!(time, x); end
        end
    end
    time = time[2:end];
    
    for l in 1:5
        y = [Bool(x) for x in labels[l]];
        X = copy(reduce(hcat, features[l][.!y])');

        gm = GMM(16, X);
        ll_no = vec(maximum(llpg(gm, X), dims=2));
        dist_no = kde(ll_no);

        X_yes = copy(reduce(hcat, features[l][y])');
        ll_yes = vec(maximum(llpg(gm, X_yes), dims=2));
        dist_yes = kde(ll_yes);
        
        plot(dist_no.x, dist_no.density, 
                color="darkgreen", linewidth=3, label="Non-Seizure");
        plot!(dist_yes.x, dist_yes.density,
                color="darkblue", linewidth=3, label="Sizure");
        savefig("$dirname/layer$l/log_like_dists.png"); closeall();

        wsize = size(features[l][1], 2);
        llmean(x) = mean(maximum(llpg(gm, copy(x')), dims=2));
        ll_no_smooth = [llmean(x) for x in features[l][.!y]];
        thresh = quantile(ll_no, 0.001);
        xpoints = (1:length(y))[.!y];
        
        valid = ll_no_smooth .>= thresh;
        ll_no_smooth = ll_no_smooth[valid];
        xpoints = xpoints[valid];
        ixs = Int.(round.(range(1,length(ll_no_smooth), length=20)));
        markers = map(x -> "$x", ixs);

        scatter(xpoints, ll_no_smooth, 
            color="green", label="", alpha=0.1,
            xticks=(ixs, markers), xrotation=45, margin=5mm,
            xlabel="Time (Seconds)", ylabel="Log-Likelihood");

        ll_yes_smooth = [llmean(x) for x in features[l][y]];
        scatter!((1:length(y))[y], ll_yes_smooth,
            color="cyan", label="", alpha=0.4);

        thresh = quantile(ll_no, 0.05);
        hline!([thresh], color="red", linewidth=3, label="");
        savefig("$dirname/layer$l/log_like_history.png"); closeall();

        time_new = Int64[];
        for s in 1:2:length(y)
            next = s == length(y) ? s-1 : s + 1;
            push!(time_new, time[next]);
        end
        time = time_new;
    end
end
