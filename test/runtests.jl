# include("restore_unregistered.jl")
# using CGCP
using POMDPs
using POMDPModels
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using Test
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
    @info "Policy value is: $(value(p,initialstate(c_tiger)))"
    (;C,p_pi) = p
    @show C
    @show p_pi
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

@testset "minihall" begin
    c_mh = MiniHallCPOMDP([4.0])

    sol = CGCPSolver(;max_iter=15,max_time=1000.0,evaluator=PolicyGraphEvaluator(),verbose=true)#PolicyGraphEvaluator()) #;method=POMDPPolicyGraphs.belief_value_recursive))
    p = solve(sol, c_mh)
    @info "Policy value is: $(value(p,initialstate(c_mh)))"
    ĉ = c_mh.constraints
    (;C,p_pi) = p
    @show C
    @show p_pi
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

@testset "cheese" begin #This doesn't match the paper
    c_mh = CheeseMazeCPOMDP([4.0])
    @show discount(c_mh)
    sol = CGCPSolver(;max_iter=15,max_time=1000.0,evaluator=PolicyGraphEvaluator()) #;method=POMDPPolicyGraphs.belief_value_recursive))
    p = solve(sol, c_mh)
    @info "Policy value is: $(value(p,initialstate(c_mh)))"
    ĉ = c_mh.constraints
    (;C,p_pi) = p
    @show C
    @show p_pi
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

@testset "maze" begin #This doesn't match the paper
    c_mz = Maze20CPOMDP([1.0])
    @show discount(c_mz)
    sol = CGCPSolver(;max_iter=15,max_time=1000.0,evaluator=PolicyGraphEvaluator(),verbose=true) #;method=POMDPPolicyGraphs.belief_value_recursive))
    p = solve(sol, c_mz)
    @info "Policy value is: $(value(p,initialstate(c_mz)))"
    ĉ = c_mz.constraints
    (;C,p_pi) = p
    @show length(C)
    @show length(p_pi)
    @info C*p_pi
    @info ĉ
    @test C*p_pi ≈ ĉ
end

include("trivial.jl")

function rolling(m,up,pol,h)
    b = initialize_belief(up,initialstate(m))
    # @show b.b
    s= 9 #rand(b)
    r_total = 0.0
    d = 1.0
    count = 0
    while !isterminal(m, s) && count < h
        count += 1
        a = action(pol, b)
        @show b.b
        @show argmax([dot(b.b,al) for al in pol.alphas])
        @show maximum([dot(b.b,al) for al in pol.alphas])
        @show s
        @show a
        sp, o, r = @gen(:sp,:o,:r)(m, s, a)
        @show o
        r_total += d*r
        b = update(up,b,a,o)
        s=sp
        d *= discount(m)
    end
    @show r_total
end