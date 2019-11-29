module TCGRN
using HDF5
using LinearAlgebra

@enum Mode direct_sum tensor_prod;
const SRATE = 256;
const DURATION_SEC = 10

export process_subject
function process_subject(subject_id)
    h5_file = h5open("../input/eeg_data_temples2.h5", "r")
    h5_node = h5_file[subject_id];
    Φ_orig, y = prepare_data(h5_node);
    close(h5_file);
    
    Φ = copy(Φ_orig);
    maxv = maximum([maximum(ϕ) for ϕ in Φ]);
    minv = minimum([minimum(ϕ) for ϕ in Φ]);
    for ix in 1:length(Φ)
        Φ[ix] = (Φ[ix] .- minv) ./ (maxv - minv);
    end

    ε = 1e-5;
    layer = 1;
    decomp_mode = Array{Mode,1}[];
    tensors = Array{Array{Float64,2}}[];
    while length(Φ) > 1
        global Φ, ε, tensors, decomp_mode, layer
        @info "Coarse graining layer: $layer"
        max_dim = maximum([size(ϕ,1) for ϕ in Φ]);
        @info "Maximum dim: $max_dim"
        U, Φ, M = coarse_grain_layer(Φ, ε);
        push!(tensors, U);
        push!(decomp_mode, M);
        layer += 1;
    end

    X = Φ[1];
    @info "Final dimension: $(size(X))"

    h5open("../temp/cgrn_$(subject_id).h5", "w") do h5_file
        write(h5_file, "X", X);
        write(h5_file, "y", y);
    end
end


function prepare_data(h5_node)
    stride = DURATION_SEC*SRATE;

    num_obs = 0;
    for obj in h5_node
        ixs = stride+1:stride:size(obj,2)
        num_obs += length(ixs);
    end

    Φ = [ones(3, num_obs) for _ in 1:stride];
    y = zeros(num_obs);
    col = 1;
    total_yes = 0.0;
    for obj in h5_node
        data = read(obj);
        total_yes += sum(data[3,:] .> 0);
        for k in stride+1:stride:size(data,2)
            slice = @view data[1:2,k-stride:k-1];
            y[col] = any(x -> x > 0, data[3,k-stride:k-1]);
            for site in 1:stride 
                Φ[site][2:3,col] = slice[:,site]; 
            end
            col += 1;
        end
    end
    @info "Total pos obs: $total_yes"
    return Φ, y
end


function coarse_grain_layer( Φ, eps )
    tensors = Array{Float64,2}[];
    features = Array{Float64,2}[];
    decomp_mode = Mode[];

    K,N = size(Φ[1]);

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        T, P, m = coarse_grain_site( Φ[ix], Φ[next], eps );
        push!(tensors, T);
        push!(features, P);
        push!(decomp_mode, m);
    end
    return (tensors, features, decomp_mode)
end


function coarse_grain_site( A::Array{Float64,2}, B::Array{Float64,2}, eps::Float64 )
    K, N = size( A );
    M = size( B, 1 );

    mode = K*M > 1024 ? direct_sum : tensor_prod
    @debug "Array size: $K x $M => Mode: $mode"

    Ω = mode == direct_sum ? accum_covmat_sum(A, B) : accum_covmat_prod(A, B);
    if sum( Ω ) == 0
        @warn "Covariance matrix is all zeros!"
        throw(TypeError("Invalid covariance matrix"));
    end

    Ω /= norm(Ω);
    λ, U = eigen( Symmetric(Ω) );
    π = sortperm(λ);
    U = U[:,π];
    λ = λ[π];
    
    if any(x -> x < -1e-5, λ)
        @error "Covariance matrix is not PSD! Eigenvalues: $(λ)"
        throw(TypeError("Covariance matrix is not PSD"))
    end

    ix = sum(λ) == 0 ? 1.0 : findall(x -> x > eps, cumsum( λ ) ./ sum( λ ))[1];
    U = U[:,ix:end];
    P = mode == direct_sum ? project_sum(U, A, B) : project_prod(U, A, B);

    return (U, P, mode)
end


function coarse_grain_data( Φ, tensors, decomp_mode )
    for ix_l in 1:length(tensors)
        ix_u = 1
        Φ¹ = Array{Float64,2}[];
        for ix in 1:2:length(Φ)
            next = ix == length(Φ) ? ix-1 : ix + 1;
            A = Φ[ix];
            B = Φ[next];
            U = tensors[ix_l][ix_u];
            mode = decomp_mode[ix_l][ix_u]; 
            ϕ = mode == direct_sum ? project_sum(U, A, B) : project_prod(U, A, B);
            push!(Φ¹, ϕ);
            ix_u += 1;
        end
        Φ = Φ¹;
    end
    return Φ
end


function accum_covmat_prod(A::Array{Float64,2}, B::Array{Float64,2})
    K,N = size( A );
    M = size( B, 1 );

    ω = zeros(K, M);
    Ω = zeros(K*M, K*M);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], ω);
        BLAS.ger!(1.0, vec(ω), vec(ω), Ω);
        fill!(ω, 0.0);
    end

    return Ω
end


function project_prod(U::Array{Float64,2}, A::Array{Float64,2}, B::Array{Float64,2})
    K, N = size( A );
    M = size( B, 1 );

    P = zeros(size(U,2), N);
    ω = zeros(K, M);
    for n in 1:N
        BLAS.ger!(1.0, A[:,n], B[:,n], ω);
        BLAS.gemv!('T', 1.0, U, vec(ω), 0.0, @view P[:,n]);
        fill!(ω, 0.0);
    end
    return P
end


function accum_covmat_sum(A::Array{Float64,2}, B::Array{Float64,2})
    K,N = size( A );
    M = size( B, 1 );

    Ω = zeros(K+M, K+M);
    for n in 1:N
        ω = [A[:,n]; B[:,n]];
        BLAS.ger!(1.0, ω, ω, Ω);
    end

    return Ω
end

function project_sum(U::Array{Float64,2}, A::Array{Float64,2}, B::Array{Float64,2})
    K, N = size( A );
    M = size( B, 1 );

    P = zeros(size(U,2), N);
    for n in 1:N
        ω = [A[:,n]; B[:,n]];
        BLAS.gemv!('T', 1.0, U, ω, 0.0, @view P[:,n]);
    end
    return P
end

println("Module loaded");
end