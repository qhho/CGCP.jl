include("../../JuliaConstrainedSARSOP.jl/test/constrained_gw.jl")
sleep_until(t) = sleep(max(t - time(), 0.0))

function main()
    po_gw = PO_GW(size=(5, 5),
        terminate_from=Set(SVector{2,Int64}[[5, 5]]),
        rewards=Dict(GWPos(5, 5) => 10.0),
        tprob=1.0)
    c_gw = ConstrainedPOMDPs.Constrain(po_gw, [2.7])
    soln = solve(CGCPSolver(), c_gw)

    #From ConstrainedSARSOP Testing
    binit = initialstate(po_gw)
    b = binit
    s = rand(initialstate(po_gw))
    d = 1.0
    r_total = 0
    up = DiscreteUpdater(po_gw)
    iter = 0
    max_fps = 10
    dt = 1 / max_fps
    a = :up
    display(render(po_gw, (sp=s, bp=b, a=a)))
    cost_tot = [0.0]
    action_list = [:up,:up,:right,:right,:right,:up,:up,:up]
    for i in 1:100
        if isterminal(po_gw, s)
            break
        end
        tm = time()
        # a = action_list[i]
        a = action(soln, b)
        prevs = s
        s, o, r = @gen(:sp, :o, :r)(po_gw, s, a)
        display("step: $iter -- s: $prevs, a: $a, sp: $s")
        r_total += d * r
        cost_tot += d * cost(c_gw, s, a)
        d *= discount(po_gw)
        b = update(up, b, a, o)
        display(render(po_gw, (sp=s, bp=b, a=a)))
        sleep_until(tm += dt)
        iter += 1
    end
    display("discounted cost: $cost_tot")
    return soln
end


soln = main();