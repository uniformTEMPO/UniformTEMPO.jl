using UniformTEMPO
using LinearAlgebra
using Test

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]

@testset "causal TEMPO: exact contraction vs brute force path sum" begin
    # without truncation the causal contraction must reproduce the Trotterized influence functional exactly
    f(t) = (1.0 + 0.3im) * exp(-abs(t) - 1.5im * t)
    bcfv(t) = [f(t) 0.4*f(t); 0.4*f(t) 0.7*f(t)]
    hs = 0.3 * sz + 0.5 * sx
    ρ0 = [0.7 0.2-0.1im; 0.2+0.1im 0.3]
    dt = 0.3
    n = 2
    u = kron(transpose(exp(im * hs * dt / 2)), exp(-im * hs * dt / 2))
    for n_c in (2, 1) # n_c = 1 (two half-step layers) tests the memory cutoff
        inf = UniformTEMPO.causal_influence([sx, sy], dt, bcfv, n_c)
        ρt = causalTEMPO(inf, 0.0, ρ0, n; h_s=hs)
        n_layers = length(inf.layer_tens)
        L = n_layers * n
        D = size(inf.layer_tens[1], 2)
        ρ_bf = zeros(ComplexF64, 4)
        for path in Iterators.product(fill(1:D, L)...)
            w = one(ComplexF64)
            for a in 1:L, b in 1:a-1
                (a - b <= inf.n_c) && (w *= inf.memory[a-b][inf.diff_index[path[a]], path[b]])
            end
            iszero(w) && continue
            v = ComplexF64.(ρ0[:])
            for a in 1:L
                isodd(a) && (v = u * v)
                v = inf.layer_tens[mod1(a, n_layers)][:, path[a], :] * v
                iseven(a) && (v = u * v)
            end
            ρ_bf .+= w .* v
        end
        @test isapprox(ρt[end][:], ρ_bf; atol=1e-12)
        @test tr(ρt[end]) ≈ 1
    end
end

@testset "causal TEMPO: results do not depend on the final time" begin
    # lines after the last time step are not stored, which must not change the dynamics at earlier times
    bcf(t) = 0.5 * exp(-abs(t) - im * t)
    ρ0 = [1.0 0; 0 0]
    ρ_short = causalTEMPO(sz, 0.1, bcf, 0.0, ρ0, 5; h_s=sx)
    ρ_long = causalTEMPO(sz, 0.1, bcf, 0.0, ρ0, 8; h_s=sx)
    @test maximum(norm(ρ_short[i] - ρ_long[i]) for i in 1:6) < 1e-12
end

@testset "causal TEMPO: spin boson agrees with uniTEMPO" begin
    bcf(t) = 0.5 * exp(-abs(t) - im * t)
    dt = 0.1
    n = 30
    ρ0 = [1.0 0; 0 0]
    pt = uniTEMPO(sz, dt, bcf, 1e-11)
    ρu = evolve(pt, ρ0, n; h_s=sx)
    ρc = causalTEMPO(sz, dt, bcf, 1e-10, ρ0, n; h_s=sx)
    @test maximum(norm(ρc[i] - ρu[i]) for i in 1:n+1) < 1e-5
end

@testset "causal TEMPO: non-commuting couplings agree with uniTEMPO" begin
    # damped Jaynes-Cummings model, Fig. 1 of arXiv:2603.23432 (weaker coupling)
    γ, Ω, g, n_bar = 2, 2, 1, 0.25
    bcf0(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcf1(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4
    dt = 0.1
    n = 10
    ρ0 = [0.5 0.5; 0.5 0.5]
    pt = uniTEMPO([sx, sy], dt, bcfv, 1e-11)
    ρu = evolve(pt, ρ0, n; h_s=0.5 * sx)
    ρc = causalTEMPO([sx, sy], dt, bcfv, 1e-8, ρ0, n; h_s=0.5 * sx)
    @test maximum(norm(ρc[i] - ρu[i]) for i in 1:n+1) < 1e-4
end

@testset "causal TEMPO: adaptive memory" begin
    # n_c = :auto drops only the layers whose memory weights are one within tol
    bcf(t) = 0.5 * exp(-3 * abs(t) - im * t)
    n = 60
    inf = UniformTEMPO.trim_memory(UniformTEMPO.causal_influence(sz, 0.1, bcf, n), 1e-8)
    @test 0 < inf.n_c < n
    ρ0 = [1.0 0; 0 0]
    ρ_full = causalTEMPO(sz, 0.1, bcf, 1e-8, ρ0, n; h_s=sx, n_c=n)
    ρ_auto = causalTEMPO(sz, 0.1, bcf, 1e-8, ρ0, n; h_s=sx)
    @test maximum(norm(ρ_full[i] - ρ_auto[i]) for i in 1:n+1) < 1e-6
end

@testset "causal TEMPO superfilter: polynomial legs and Taylor gates" begin
    # complete bases (degree 2 spans all functions of the labels for one spin, D = 4, D_diff = 3) and a high Taylor
    # order reproduce the full contraction; the default (degree 2, Taylor order 2) is close
    bcf(t) = 0.5 * exp(-abs(t) - im * t)
    ρ0 = ComplexF64[1 0; 0 0]
    full = causalTEMPO(sz, 0.1, bcf, 0.0, ρ0, 12; h_s=sx, n_c=12)
    exact = causalTEMPO_superfilter(sz, 0.1, bcf, 0.0, ρ0, 12; h_s=sx, n_c=12, taylor_order=16)
    @test maximum(norm(full[i] - exact[i]) for i in 1:13) < 1e-12
    default = causalTEMPO_superfilter(sz, 0.1, bcf, 0.0, ρ0, 12; h_s=sx, n_c=12)
    @test maximum(norm(full[i] - default[i]) for i in 1:13) < 1e-4

    # non-commuting couplings and two bath channels (damped Jaynes-Cummings, D = 16, D_diff = 9): complete at degree 4,
    # which tests the boundary with the Trotter-ordered projections and both time-local weights
    γ, Ω, g, n_bar = 2, 2, 1, 0.25
    bcf0(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcf1(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4
    ρ0 = ComplexF64[0.5 0.5; 0.5 0.5]
    full = causalTEMPO([sx, sy], 0.1, bcfv, 0.0, ρ0, 6; h_s=0.5 * sx, n_c=6)
    exact = causalTEMPO_superfilter([sx, sy], 0.1, bcfv, 0.0, ρ0, 6; h_s=0.5 * sx, n_c=6, mu_degree=4, line_degree=4, taylor_order=16)
    @test maximum(norm(full[i] - exact[i]) for i in 1:7) < 1e-12
    default = causalTEMPO_superfilter([sx, sy], 0.1, bcfv, 0.0, ρ0, 6; h_s=0.5 * sx, n_c=6)
    @test maximum(norm(full[i] - default[i]) for i in 1:7) < 1e-4

    # the memory cutoff without the memory weights agrees with trim_memory
    bcf3(t) = 0.5 * exp(-3 * abs(t) - im * t)
    for (s, b) in ((sz, bcf3), ([sx, sy], bcfv))
        inf = UniformTEMPO.trim_memory(UniformTEMPO.causal_influence(s, 0.1, b, 60), 1e-8)
        @test UniformTEMPO.superfilter_influence(s, 0.1, b, 60; trim=1e-8).n_c == inf.n_c
    end

    # two RWA emitters (R = 15 of D = 256, R_d = 16 of D_diff = 81): trace kept, decoupled second emitter
    k0(t) = 0.1 * exp(-abs(t) - 0.3im * t)
    B = [1 -im; im 1] / 4
    I2 = Matrix{ComplexF64}(I, 2, 2)
    sops = [kron(sx, I2), kron(sy, I2), kron(I2, sx), kron(I2, sy)]
    ρ0 = zeros(ComplexF64, 4, 4); ρ0[2, 2] = 1
    sf = UniformTEMPO.superfilter_influence(sops, 0.1, t -> [B*k0(t) 0.5*B*k0(t); 0.5*B*k0(t) B*k0(t)], 20)
    @test (length(sf.c_one), length(sf.line_one)) == (15, 16)
    ρt = causalTEMPO_superfilter(sf, 1e-10, ρ0, 20; h_s=0.05 * kron(sx, I2))
    @test maximum(abs(tr(ρ) - 1) for ρ in ρt) < 1e-8
    single = causalTEMPO_superfilter([sx, sy], 0.1, t -> B * k0(t), 1e-10, ComplexF64[1 0; 0 0], 20; h_s=0.05 * sx)
    pair = causalTEMPO_superfilter(sops, 0.1, t -> [B*k0(t) zeros(2, 2); zeros(2, 2) B*k0(t)], 1e-10, ρ0, 20; h_s=0.05 * kron(sx, I2))
    na = kron([1 0; 0 0], I2)
    @test maximum(abs(real(single[i][1, 1]) - real(tr(na * pair[i]))) for i in 1:21) < 1e-6
end
