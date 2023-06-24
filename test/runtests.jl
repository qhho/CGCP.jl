# include("restore_unregistered.jl")
# using CGCP
using POMDPs
using POMDPModels
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using Test
using SARSOP
using LinearAlgebra

@testset "rock" begin
    ĉ = [1.0]
    c_rs = RockSampleCPOMDP()

    sol = CGCPSolver(;max_time=10.0,verbose=true)
    p = solve(sol, c_rs)
    (;C,p_pi) = p
    @test C*p_pi ≈ ĉ
end

@testset "tiger" begin
    ĉ = [1.0]

    c_tiger = constrain(TigerPOMDP(), ĉ) do s,a
        iszero(a) ? [1.0] : [0.0]
    end

    sol = CGCPSolver()
    p = solve(sol, c_tiger)
    (;C,p_pi) = p
    @test C*p_pi ≈ ĉ
end

include("trivial.jl")
