module TCGRNUtils
    using Images
    using Profile
    using ITensors
    using TestImages
    using Statistics
    using MLDatasets
    using Distributed
    using ScikitLearn
    using LinearAlgebra

    const EPS = 1e-3;

    export coarse_grain_site

    function coarse_grain_site( A, B, ix )
        @info "Coarse graining site: $ix"
        s1 = findindex(inds(A), "site");
        s2 = findindex(inds(B), "site");

        A¹ = prime(A, s1);
        B¹ = prime(B, s2);
        
        Σ = (A*B)*(A¹*B¹);
        Σ /= norm(Σ);
        U,Λ,Uᵗ = svd(Σ, s1, s2, cutoff=EPS);
        ϕ = broadcast_operator( A, B, U );
        return U, ϕ, ix
    end


    function broadcast_operator(A, B, U)
        s1 = findindex(inds(A), "site");
        s2 = findindex(inds(B), "site");
        obs_index = findindex(A, "obs");
        N = dim(obs_index);

        α = ITensor(commoninds(A, U));
        β = ITensor(commoninds(B, U));
        
        ixᵤ = findindex(inds(U), "Link,u")

        # Danger!! This requires a specific ordering of the indexes
        getix_a(n) = ((n-1)*dim(s1))+1;
        getix_b(n) = ((n-1)*dim(s2))+1;
        data = zeros(dim(ixᵤ), N);
        for n in 1:N
            ixa = getix_a(n);
            ixb = getix_b(n);
            α.store.data[:] = A.store.data[ixa:ixa+dim(s1)-1];
            β.store.data[:] = B.store.data[ixb:ixb+dim(s2)-1];
            γ = α*U*β;
            data[:,n] = γ.store.data;
        end
        
        site_index = Index(dim(ixᵤ), "site");
        ϕ = ITensor(site_index, obs_index);
        ϕ.store.data[:] = data[:];
        return ϕ
    end
end