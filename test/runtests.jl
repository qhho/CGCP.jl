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
    c_rs = RockSampleCPOMDP()

    sol = CGCPSolver(;max_time=60.0,verbose=true,evaluator=PolicyGraphEvaluator())
    p = solve(sol, c_rs)
    ĉ = c_rs.constraints
    (;C,p_pi) = p
    @show C
    @show p_pi
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

@testset "tiger" begin
    ĉ = [1.0]

    c_tiger = constrain(TigerPOMDP(), ĉ) do s,a
        iszero(a) ? [1.0] : [0.0]
    end

    sol = CGCPSolver(;verbose=true,evaluator=PolicyGraphEvaluator())
    p = solve(sol, c_tiger)
    (;C,p_pi) = p
    @show C
    @show p_pi
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

include("trivial.jl")