using DSP
using HDF5
using Plots
using Random
using MLKernels
using StatsBase
using DataFrames
using Statistics
using Clustering
using Distributed
using KernelDensity
using Interpolations
using GaussianMixtures
using GaussianProcesses
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

    TCGRN.process_subject(subject_ids[1], 1e-7);
    features, labels = coarse_grain_data(subject_ids[1], 1e-7);    

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
            xticks=(ixs, markers), xrotation=45,
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


function layer_analysis_gauss(features, labels)
    for l in 3:5
        y = [Bool(x) for x in labels[l]];

        val = findfirst(x -> x > 0, y);
        divergences = zeros(val, val);
        means, covmats = compute_gauss_dists(features[l][1:val]);
        off_diag_covar = [sum(diag(Σ)) / sum(Σ) for Σ in covmats];
        stats = describe(off_diag_covar);

        K = size(features[l][1],1);
        for i in 1:val
            for j in (i+1):val
                @info "KL => $i,$j"
                @inbounds Σ₁, Σ₂ = diag(covmats[i]), diag(covmats[j]);
                @inbounds μ₁, μ₂ = means[i], means[j];
                @time wass = compute_wasserstein_diag(μ₁, μ₂, Σ₁, Σ₂);
                divergences[i,j] = wass;
            end
        end
        
        # Some spectral clustering to see if we can merge some distributions...
        Δ = convert( Array{Float64,2}, Symmetric( divergences ) ) .^ -1;
        Δ[ Δ .== Inf ] .= 0.0;
        W = zeros( size( Δ ) );
        W[ diagind(W) ] = sum( Δ, dims=2 );

        λ, U = eigen( W .- Δ, W );
        U = U[:,end-64:end];
        km = kmeans( U', 64 );

        # Now assemble a database of samples to use for analyzing MMD
        dictionary = Array{Float64,2}[];
        for c in unique(km.assignments)
            X = reduce(hcat, features[l][1:val][ km.assignments .== c ]);
            N = min(5000, size(X,2));
            X = X[:,shuffle(1:size(X,2))[1:N]];
            push!(dictionary, X);
        end

        mmd_vals = Array{Float64,1}[];
        for X in features[l]
            @time mmd = [compute_mmd(V, X) for V in dictionary];
            push!(mmd_vals, mmd);
        end

    end
end


function compute_kldiv(μ₁, μ₂, Σ₁, Σ₂, K)
    Σ₂_inv = inv( Σ₂ );
    kl = 0.5*(tr(Σ₂_inv*Σ₁) + 
           ((μ₂ .- μ₁)'*Σ₂_inv*(μ₂ .- μ₁))[1] - K + logdet(Σ₂) - logdet(Σ₁));
    return kl
end


function compute_kldiv_diag(μ₁, μ₂, Σ₁, Σ₂, K)
    Σ₂_inv = Σ₂.^-1;
    μ_diff = μ₁ .- μ₂;
    kl = 0.5*(sum(Σ₂_inv .* Σ₁) + 
              ((μ_diff'.*Σ₂_inv)*μ_diff)[1] - K + 
              log(prod(Σ₂)) - log(prod(Σ₁)))
    return kl
end


function compute_wasserstein_diag(μ₁, μ₂, Σ₁, Σ₂)
    wass = norm(μ₁ - μ₂)^2 + sum(Σ₁ .+ Σ₂ - 2*sqrt.(sqrt.(Σ₂) .* Σ₁ .* sqrt.(Σ₂)))
    return wass
end


function compute_gauss_dists(features)
    N = size(features[1],2);
    means = Array{Float64,2}[];
    covmats = Array{Float64,2}[];
    for X in features
        Σ = (1/N)*(X*X');
        μ = mean(X, dims=2);
        push!(means, μ);
        push!(covmats, Σ);
    end
    return means, covmats
end


function compute_mmd(X, Y)
    Kxx = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), X, X));
    Kxy = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), X, Y));
    Kyy = sum(kernelmatrix(Val(:col), SquaredExponentialKernel(), Y, Y));

    N = size(X,2);
    M = size(Y,2);

    return (1/(N^2))*Kxx + ((1/(M^2)))*Kyy - (2/(M*N))*Kxy; 
end


function compute_linear_mmd(X, Y)
    mmd = 0.0
    N = min(size(X,2), size(Y,2));
    ixx = shuffle(1:size(X,2))[1:N];
    ixy = shuffle(1:size(Y,2))[1:N];
    for ix in 1:N
        mmd += gauss_kern(vec( X[:,ixx[ix]] ), vec( Y[:,ixy[ix]] ), 1.0, 1.0);
    end

    mmd *= (1/N)
    return mmd
end

function gauss_kern(x::Array{Float64,1}, 
                    y::Array{Float64,1}, 
                    σ::Float64, l::Float64)::Float64
    return (σ^2)*exp(-(norm(x .- y)^2) / (2*(l^2)))
end

