using HDF5
using Statistics
using Distributed
using LinearAlgebra

@everywhere include("tcgrn_utils.jl")
@everywhere using .TCGRN

function main()
    h5_file = h5open("../input/eeg_data_temples2.h5")
    subject_ids = names(h5_file);
    close(h5_file)

    pmap(process_subject, subject_ids);
end
