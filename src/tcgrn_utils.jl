module TCGRN
using HDF5
using Statistics
using LinearAlgebra

@enum Mode direct_sum tensor_prod;
const SRATE = 256;
const DURATION_SEC = 2
const ZERO_THRESH = 1e-10;

export process_subject, prepare_data, Mode, rescale_layer!
function process_subject(subject_id, ε)
    println("Opening data file");
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5", "r");
    h5_node = h5_file[subject_id];
    println("Preparing data for subject $subject_id");
    Φ = prepare_data(h5_node);
    close(h5_file);

    println("Positive fraction: $(mean([S.y for S in Φ]))")

    layer = 1;
    if !isdir("../temp/$(ε)")
        mkpath("../temp/$(ε)");
    end
    
    HDF5.h5open("../temp/$(ε)/decomp_$subject_id.h5", "w") do h5_file
        while length(Φ) > 1
            println("Coarse graining layer: $layer");
            max_dim = maximum([size(S.ϕ,1) for S in Φ]);
            max_val = maximum([maximum(S.ϕ) for S in Φ]);
            min_val = minimum([minimum(S.ϕ) for S in Φ]);
            Φ = coarse_grain_layer(Φ, 1e-5);
            println(
                join(["Maximum dimension: $max_dim", 
                      "Num sites: $(length(Φ))", 
                      "Positive fraction: $(mean([S.y for S in Φ]))",
                      "Maximum: $max_val", "Minimum: $min_val"],
                      "\n"));
            write_layer(Φ, layer, h5_file);
            layer += 1;
        end
    end
end


function write_layer(Φ, layer_num, h5_file)
    num_sites = length(Φ);
    h5_node = HDF5.g_create(h5_file, "layer$layer_num");
    for s in 1:num_sites
        site_node = HDF5.g_create(h5_node, "site$s")
        write(site_node, "U", Φ[s].U);
        write(site_node, "mode", Int(Φ[s].M));
        write(site_node, "eigvals", Φ[s].λ);
    end
end


struct SiteInfo
    U::Array{Float64,2};
    λ::Array{Float64,1};
    ϕ::Array{Float64,2};
    M::Mode;
    y::Bool;
end


function prepare_data(h5_node)
    stride = 2*SRATE;
    seg_length = DURATION_SEC*SRATE;

    Φ = SiteInfo[];

    minv, maxv = get_scaler(h5_node);
    for obj in h5_node
        data = read(obj);
        for k in seg_length+1:seg_length:size(data,2)
            y = any(x -> x > 0, data[3,k-seg_length:k-1]);
            ϕ = ones(4, seg_length);
            ϕ[2:3,:] = (data[1:2,k-seg_length:k-1] .- minv) ./ (maxv - minv);
            ϕ[4,:] = ϕ[2,:] .* ϕ[3,:];
            S = SiteInfo(zeros(2,2), zeros(2), ϕ, direct_sum, y);
            push!(Φ, S);
        end
    end

    return Φ
end


function get_scaler( h5_node )
    h5_file = HDF5.h5open("../temp/bounds.h5", "cw");
    subject_id = replace(HDF5.name(h5_node), "/" => "");
    if !(subject_id in names(h5_file))
        minv = Float64[];
        maxv = Float64[];
        for obj in h5_node
            data = read(obj)[1:2,:]
            push!(minv, minimum(data))
            push!(maxv, maximum(data));
        end
        group = HDF5.g_create(h5_file, subject_id);
        write(group, "minv", minimum(minv));
        write(group, "maxv", maximum(maxv));
    end
    g = h5_file["$subject_id"];
    minv = read(g["minv"]);
    maxv = read(g["maxv"]);
    close(h5_file);
    return minv, maxv
end


function coarse_grain_layer( Φ::Array{SiteInfo,1}, eps::Float64 )
    features = Array{Float64,2}[];
    y_new = Bool[];
    L = SiteInfo[];

    K,N = size(Φ[1].ϕ);

    for ix in 1:2:length(Φ)
        next = ix == length(Φ) ? ix-1 : ix + 1;
        T, P, m, λ = coarse_grain_site( Φ[ix].ϕ, Φ[next].ϕ, eps );
        S = SiteInfo(T, λ, P, m, Φ[ix].y | Φ[next].y);
        push!(L, S)
    end

    return L
end


function coarse_grain_site( A::Array{Float64,2}, B::Array{Float64,2}, eps::Float64 )
    K, N = size( A );
    M = size( B, 1 );

    mode = K*M > 1024 ? direct_sum : tensor_prod
    @debug "Array size: $K x $M => Mode: $mode"

    Ω = mode == direct_sum ? accum_covmat_sum( A, B ) : accum_covmat_prod( A, B );
    Ω /= norm( Ω );
    λ, U = eigen( Symmetric( Ω ) );
    π = sortperm( λ );
    U = U[:,π];
    λ = λ[π];
    if sum(λ) == 0
        @warn "Eigenvalues were all zero!"
        @show Ω
        @show λ
        @show π
        show(IOContext(stdout, :limit => true), "text/plain", A);
        show(IOContext(stdout, :limit => true), "text/plain", B);
        λ[end] = 1.0;
    end
    
    if any(x -> x < -1e-5, λ)
        @error "Covariance matrix is not PSD! Eigenvalues: $(λ)"
        throw(TypeError("Covariance matrix is not PSD"))
    end

    errors = findall(x -> x > eps, cumsum( λ ) ./ sum( λ ));
    ix = length(errors) > 0 ? errors[1] : length(λ);
    U = U[:,ix:end];
    P = mode == direct_sum ? project_sum(U, A, B) : project_prod(U, A, B);
    P[abs.(P) .< ZERO_THRESH] .= 0.0;

    return (U, P, mode, λ)
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
