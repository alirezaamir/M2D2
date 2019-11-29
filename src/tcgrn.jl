using HDF5
using Random
using Distributed
using Statistics

addprocs(10, enable_threaded_blas=true);

@everywhere using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN


function main()
    h5_file = h5open("../input/eeg_data_temples2.h5")
    subject_ids = names(h5_file);
    close(h5_file)

    pmap(process_subject, subject_ids);
end


function analysis(subject_id)
    X = h5read("../temp/cgrn_$(subject_id).h5", "/X");
    y = h5read("../temp/cgrn_$(subject_id).h5", "y");

    ixs = sort(rand((1:size(X,2))[y .== 0], 2000));
    X_no = X[:,ixs];
    X_yes = X[:, y .== 1];

    mmd = compute_mmd(X_yes, X_no);
    @info "MMD (between types): $mmd"

    ixs = sort(rand((1:size(X,2))[y .== 0], 2000));
    X_no2 = X[:,ixs];
    
    mmd = compute_mmd(X_no, X_no2);
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