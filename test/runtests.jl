include("restore_unregistered.jl")
using CGCP
using POMDPs
using POMDPModels
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using Test
using SARSOP
using LinearAlgebra

@testset "tiger" begin
    ĉ = [1.0]
    c_tiger = Constrain(TigerPOMDP(), ĉ)
    ConstrainedPOMDPs.cost(m::typeof(c_tiger), s, a) = iszero(a) ? [1.0] : [0.0]

    sol = CGCPSolver()
    p = solve(sol, c_tiger)
    (;C,p_pi) = p
    @test C*p_pi ≈ ĉ
end

include("trivial.jl")
