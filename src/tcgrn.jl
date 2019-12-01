using HDF5
using Random
using Distributed
using Statistics

nproc = min(10, Int(length(Sys.cpu_info()) / 8));
addprocs(nproc, enable_threaded_blas=true);

@everywhere using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN


function main()
    h5_file = HDF5.h5open("../input/eeg_data_temples2.h5")
    subject_ids = HDF5.names(h5_file);
    close(h5_file)
    #show subject_ids

    ε = 1e-9*ones(length(subject_ids));
    pmap(TCGRN.process_subject, subject_ids, ε);
end


function analysis(subject_id)
    X = HDF5.h5read("../temp/cgrn_$(subject_id).h5", "/X");
    y = HDF5.h5read("../temp/cgrn_$(subject_id).h5", "y");

    X_no = X[:, y .== 0];
    X_yes = X[:, y .== 1];

    mmd = compute_mmd(X_yes, X_no);
    @info "MMD (within type): $mmd"
end


function compute_mmd(X::Array{Float64,2}, Y::Array{Float64,2})
    Kxx = @spawnat :any kernsum(X);
    Kyy = @spawnat :any kernsum(Y);
    Kxy = @spawnat :any kernsum(X,Y);

    N = size(X,2);
    M = size(Y,2);
    mmd = (2.0/(N*(N-1)))*fetch(Kxx) + 
          (2.0/(M*(M-1)))*fetch(Kyy) + 
          (2.0/(M*N))*fetch(Kxy);
    return mmd 
end


@everywhere function kernsum(X::Array{Float64,2})
    K = 0.0;
    for i in 1:size(X,2)
        for j in i+1:size(X,2)
            K += gauss_kern(X[:,i], X[:,j], 1.0);
        end
    end
    return K
end


@everywhere function kernsum(X::Array{Float64,2}, Y::Array{Float64,2})
    K = 0.0
    for x in 1:size(X,2)
        for y in 1:size(Y,2)
            K += gauss_kern(X[:,x], Y[:,y], 1.0);
        end
    end
    return K
end


@everywhere function gauss_kern(x::Array{Float64,1}, y::Array{Float64,1}, γ)
    return exp(-γ*norm(x .- y)^2)
end

main()

