using POMDPs
using ConstrainedPOMDPModels
using StaticArrays
using ConstrainedPOMDPs

# sleep_until(t) = sleep(max(t - time(), 0.0))

# function POMDPs.reward(m::ConstrainedPOMDPModels.GridWorldPOMDP, s, a)
#     po2_gw = ConstrainedPOMDPModels.GridWorldPOMDP(size=(5, 5),
#     terminate_from=Set(SVector{2,Int64}[[5, 5]]),
#     rewards=Dict(ConstrainedPOMDPModels.GWPos(5, 5) => 10.0),
#     tprob=1.0)
#     c2_gw = ConstrainedPOMDPs.Constrain(po2_gw, [1.0])
#     return reward(m.mdp, s, a) - [1]'*cost(c2_gw, s, a)
# end

# function ConstrainedPOMDPs.cost(constrained::ConstrainedPOMDPModels.ConstrainedGridWorld, s, a)
#     return [0.0]
# end

function main()
    println("Instantiate Problem") 
    po_gw = ConstrainedPOMDPModels.GridWorldPOMDP(size=(5, 5),
        terminate_from=Set(SVector{2,Int64}[[5, 5]]),
        rewards=Dict(ConstrainedPOMDPModels.GWPos(5, 5) => 10.0),
        tprob=1.0)
    c_gw = ConstrainedPOMDPs.Constrain(po_gw, [1.0])

    println("Solve")
    soln = solve(CGCPSolver(;h=6), c_gw)
    return soln
    @show soln.h

    # println("More Testing")    
    # #From ConstrainedSARSOP Testing
    # binit = initialstate(po_gw)
    # b = binit
    # s = rand(initialstate(po_gw))
    # d = 1.0
    # r_total = 0
    # up = DiscreteUpdater(po_gw)
    # iter = 0
    # max_fps = 10
    # dt = 1 / max_fps
    # a = :up
    # display(render(po_gw, (sp=s, bp=b, a=a)))
    # cost_tot = [0.0]
    # action_list = [:up,:up,:right,:right,:right,:up,:up,:up]
    # for i in 1:100
    #     if isterminal(po_gw, s)
    #         break
    #     end
    #     tm = time()
    #     # a = action_list[i]
    #     a = action(soln, b)
    #     prevs = s
    #     s, o, r = @gen(:sp, :o, :r)(po_gw, s, a)
    #     display("step: $iter -- s: $prevs, a: $a, sp: $s")
    #     r_total += d * r
    #     cost_tot += d * cost(c_gw, s, a)
    #     d *= discount(po_gw)
    #     b = update(up, b, a, o)
    #     display(render(po_gw, (sp=s, bp=b, a=a)))
    #     sleep_until(tm += dt)
    #     iter += 1
    # end
    # display("discounted cost: $cost_tot")
    # return soln
end


soln = main();

# ### CGCP Struct Test
# po_gw = ConstrainedPOMDPModels.GridWorldPOMDP(size=(5, 5),
#         terminate_from=Set(SVector{2,Int64}[[5, 5]]),
#         rewards=Dict(ConstrainedPOMDPModels.GWPos(5, 5) => 10.0),
#         tprob=1.0)
# c_gw = ConstrainedPOMDPs.Constrain(po_gw, [1.0])

# # function ConstrainedPOMDPs.cost(constrained::ConstrainedPOMDPModels.ConstrainedGridWorld, s, a)
# #     return [0.0]
# # end

# M1 = CGCP.CGCPProblem(c_gw,ones(length(c_gw.constraints)),true,6)
# pol1 = CGCP.compute_policy(M1, ones(length(c_gw.constraints)), 0.0, 0.0)

# PG_reward(m::ConstrainedPOMDPModels.GridWorldPOMDP, s, a, sp) =  PG_reward(m, s, a)

# function PG_reward(m::ConstrainedPOMDPModels.GridWorldPOMDP,s,a)
#     return [reward(m, s, a), ConstrainedPOMDPs.cost(c_gw,s,a)...]
# end

# pol2 = PBVI.solve(PBVISolver(max_iter=10),po_gw)
# # pol3 = PBVI.solve(PBVISolver(max_iter=10),po_gw)
# #Call PBVI Twice
# #Test w/ SARSOP
# #Check PG construction with same policy and different problem
# up1 = DiscreteUpdater(po_gw)
# b01 = initialize_belief(up1,initialstate(po_gw))
# testpg = BeliefValue(po_gw,up1,pol2,b01,6;rewardfunction=PG_reward)

# up2 = DiscreteUpdater(M1)
# b02 = initialize_belief(up2,initialstate(M1))
# testpg2 = BeliefValue(M1,up2,pol2,b02,6;rewardfunction=CGCP.PG_reward)