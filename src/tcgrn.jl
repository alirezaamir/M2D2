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

    for subject_id in subject_ids
        ε_list = [0.5, 1e-1, 1e-2, 1e-3, 1e-4];
        for ε in ε_list
            TCGRN.process_subject(subject_ids[1], 1e-7);
            features, labels = coarse_grain_data(subject_ids[1], 1e-7);
            dirname = "../output/$subject_id/$ε"
            if !isdir(dirname)
                mkpath(dirname);
            end

            mmd_between, mmd_within = layer_analysis_mmd(features, labels);
            
            open("$dirname/mmd_between.txt", "w") do fh
                write(fh, "mmd,se,N\n");
                for (mmd, se, N) in mmd_between write(fh, "$mmd,$se,$N\n"); end
            end
            open("$dirname/mmd_within.txt", "w") do fh
                write(fh, "mm-d,se,N\n");
                for (mmd, se, N) in mmd_within write(fh, "$mmd,$se,$N\n"); end
            end
        end
    end
end


function plot_eigenvalues(subject_id, labels, ε)
    HDF5.h5open("../temp/$ε/decomp_$(subject_id).h5") do h5_file
        for l in 1:length(labels)
            layer_node = h5_file["layer$l"];
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
            outpath = "../output/$subject_id/$ε/layer$l"
            if !isdir(outpath)
                mkpath(outpath)
            end
            savefig("$outpath/eigenvals.png");
        end
    end
end


function layer_analysis_cancor(features, labels, outdir)
    X_max = features[end][1];
    corrs = Array{Array{Float64,1},1}[];
    used_labels = Array{Bool,1}[];
    for l in 1:min(5, length(features))
        @info "Layer $l/$(length(features))"
        layer_corr = Array{Float64,1}[];

        for s in 1:length(features[l])
            X_l = features[l][s];
            @time C = fit(CCA, X_max, X_l);
            push!(layer_corr, correlations(C));
        end
        push!(corrs, layer_corr);
    end

    for l in 1:5
        dim = maximum([length(x) for x in corrs[l]]);
        M = zeros(dim, length(corrs[l]));
        avg_corr = zeros(length(corrs[l]));
        max_corr = zeros(length(corrs[l]));
        min_corr = zeros(length(corrs[l]));
        for ix in 1:length(corrs[l])
            M[1:length(corrs[l][ix]),ix] = corrs[l][ix];
            avg_corr[ix] = mean(corrs[l][ix]);
            max_corr[ix] = maximum(corrs[l][ix]);
            min_corr[ix] = minimum(corrs[l][ix]);
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

        outpath = "$outdir/layer$l"
        if !isdir(outpath)
            mkpath(outpath)
        end
        savefig("$outpath/correlates.png"); closeall();

        plot(1:3:length(avg_corr), avg_corr[1:3:length(avg_corr)], 
             linewidth=2, color="darkgreen", label="Mean");
        plot!(1:3:length(avg_corr), min_corr[1:3:length(avg_corr)], 
             linewidth=2, color="darkblue", label="Min");
        plot!(1:3:length(avg_corr), max_corr[1:3:length(avg_corr)], 
             linewidth=2, color="orange", label="Max");

        ix = 1;
        start = -1;
        flag = false;
        y = labels[l] .> 0;
        while ix < length(y)
            if y[ix] & (start < 0)
                start = ix;
            end
            if !y[ix] & (start > 0)
                @info "Seizure from: $start -> $ix"
                vspan!([start, ix], color="red", label="");
                start = -1;
            end
            ix += 1;
        end
        savefig("$outpath/ts_corr.png"); closeall();
    end
end


function logit(features, labels)

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


function layer_analysis_mmd(features, labels)
    mmd_between = Tuple{Float64,Float64,Int64}[];
    mmd_within = Tuple{Float64,Float64,Int64}[];
    for l in 1:min(5, length(features))
        @info "Layer $l"
        X = features[l];
        y = map(x -> x > 0, labels[l]);

        if length(y) == sum(y)
            continue
        end 

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

        push!(mmd_between, (mean(between), std(between), sum(y)));
        push!(mmd_within, (mean(within), std(within), sum(y)));

        @info "Layer $l => MMD (Between): $(mean(between)) ($(std(between)))"
        @info "Layer $l => MMD (Within): $(mean(within)) ($(std(within)))"
    end

    return mmd_between, mmd_within
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


# main()
