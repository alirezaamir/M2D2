using HDF5
using Plots
using Random
using MLKernels
using StatsBase
using Statistics
using Distributed
using MultivariateStats


nproc = min(10, Int(length(Sys.cpu_info()) / 8));
addprocs(nproc, enable_threaded_blas=true);

@everywhere using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN


function main()
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5")
    subject_ids = HDF5.names(h5_file);
    close(h5_file)

    ε_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
    for ε in ε_list
        TCGRN.process_subject(subject_ids[1], ε);
        features, labels = coarse_grain_data(subject_ids[1], 1e-5);
    end
end


function plot_eigenvalues(subject_id)
    HDF5.h5open("../temp/decomp_$(subject_id).h5") do h5_file
        for (l, layer_node) in enumerate(h5_file)
            eigs = Array{Float32,1}[];
            for site_node in layer_node
                push!(eigs, read(site_node, "eigvals"));  
            end

            maxv = maximum([length(x) for x in eigs]);
            λ = fill!(zeros(maxv, length(eigs)), NaN);
            for ix in 1:length(eigs)
                λ[1:length(eigs[ix]),ix] = reverse(
                    cumsum(eigs[ix]) ./ sum(eigs[ix]));
            end
            nanmean(x) = mean(filter(!isnan, x));
            nanstd(x) = std(filter(!isnan, x));
            μ = mapslices(nanmean, λ, dims=2);
            σ = map(x -> isnan(x) ? 0.0 : x, mapslices(nanstd, λ, dims=2))

            which = [ix for ix in 1:length(μ) if μ[ix] > 1e-5];
            plot(which, vec(μ[which]), 
                 linewidth=3, grid=false, yerror=vec(σ[which]), label="");
            outpath = "../output/$subject_id/layer$l"
            if !isdir(outpath)
                mkpath(outpath)
            end
            savefig("$outpath/eigvals_level$l.png");
        end
    end
end


function layer_analysis_cancor(features, labels)
    X_max = features[end][1];
    corrs = Array{Array{Float64,1},1}[];
    for l in 1:length(features)
        layer_corr = Array{Float64,1}[];
        for s in 1:length(features[l])
            X_l = features[l][s];
            C = fit(CCA, X_max, X_l);
            push!(layer_corr, correlations(C));
        end
        push!(corrs, layer_corr);
    end

    for l in 1:length(corrs)
        dim = maximum([length(x) for x in corrs[l]]);
        M = zeros(dim, length(corrs[l]));
        avg_corr = zeros(length(corrs[l]));
        for ix in 1:length(corrs[l])
            M[1:length(corrs[l][ix]),ix] = corrs[l][ix];
            avg_corr = mean(corrs[l][ix]);
        end
        
        inds = [Bool(x) for x in labels[l]];
        μ_yes = mean(M[:,inds], dims=2);
        σ_yes = std(M[:,inds], dims=2);
        μ_no = mean(M[:,.!inds], dims=2);
        σ_no = std(M[:,.!inds], dims=2);

        inds = μ_yes .> 0.70;
        μ_yes = μ_yes[inds];
        σ_yes = σ_yes[inds];

        inds = μ_no .> 0.70;
        μ_no = μ_no[inds];
        σ_no = σ_no[inds];

        plot(1:length(μ_yes), μ_yes, 
             linewidth=2, 
             yerror=σ_yes, 
             color="darkgreen",
             label="Seizure");
        plot!(1:length(μ_no), μ_no,
              linewidth=2,
              yerror=σ_no,
              color="darkblue",
              label="Non-Seizure");
        xlabel!("Canonical Correlate");
        ylabel!("Correlation");
        title!("Resolution (Seconds): $((2^l)*10)");

        outpath = "../output/$subject_id/layer$l"
        if !isdir(outpath)
            mkpath(outpath)
        end
        savefig("$outpath/correlates.png");
        
        plot(avg_corr, linewidth=3, color="darkgreen")

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
    for s in 1:length(names(layer_node))
        next = ix == length(X) ? ix-1 : ix + 1;
        site_node = layer_node["site$s"];
        mode = TCGRN.Mode(read(site_node["mode"]));
        U = read(site_node["U"]);
        A = X[ix] / norm(X[ix]); B = X[next] / norm(X[next]);
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


function layer_analysis_mmd(subject_id)
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5", "r");
    h5_node = h5_file[subject_id];
    Φ = prepare_data(h5_node);
    close(h5_file);

    X = [S.ϕ for S in Φ];
    y = [S.y for S in Φ];
    mmd_within = Tuple{Float64,Float64}[];
    mmd_between = Tuple{Float64,Float64}[];
    h5open("../temp/decomp_$subject_id.h5") do h5_file
        num_layers = length(names(h5_file));
        for l in 1:num_layers
            # global X, y, mmd_within, mmd_between;
            layer_node = h5_file["layer$l"];
            X, y = predict_layer(X, y, layer_node);
            Φ_yes = copy(reduce(vcat, X[y])');

            N = min(3*sum(y), length(y) - sum(y));
            between = Float64[];
            within = Float64[];

            for _ in 1:100
                samp = sample((1:length(y))[.!y], N, replace=false);
                Φ_no = copy(reduce(vcat, X[samp])');
                push!(between, compute_mmd(Φ_yes, Φ_no));

                samp = sample((1:length(y))[.!y], N, replace=false);
                Φ_no2 = copy(reduce(vcat, X[samp])');
                push!(within, compute_mmd(Φ_no2, Φ_no));
            end

            push!(mmd_between, (mean(between), std(between)));
            push!(mmd_within, (mean(within), std(within)));

            @info "Layer $l => MMD (Between): $(mean(between)) ($(std(between)))"
            @info "Layer $l => MMD (Within): $(mean(within)) ($(std(within)))"
        end
    end
end


function compute_mmd(X::Array{Float64,2}, Y::Array{Float64,2})
    Kxx = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), X));
    Kyy = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), Y));
    Kxy = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), X, Y));

    N = size(X,2);
    M = size(Y,2);
    mmd = (1.0/(N*N))*fetch(Kxx) + 
          (1.0/(M*M))*fetch(Kyy) -
          (2.0/(M*N))*fetch(Kxy);
    return mmd 
end


main()

