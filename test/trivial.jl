@testset "trivial" begin
    ĉ = [0.5]
    pomdp = TrivialCPOMDP(ĉ)
    sol = CGCPSolver(
        pomdp_solver = SARSOPSolver(verbose=false, precision=1e-2)
    )
    p = solve(sol, pomdp)
    
    @test sum(p.p_pi) ≈ 1.0
    @test dot(p.p_pi, p.V) ≈ only(ĉ)
    @test p.C*p.p_pi ≈ ĉ
    @test all(
        all(λ .≥ 0) for λ ∈ p.dual_vectors
    )
end
