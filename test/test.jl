using JuMP
import GLPK


function dostuff()
    m = Model(GLPK.Optimizer)

    @variable(m, x[1:3] >= 0)

    V = [3.0,4.0,5.0]
    @objective(
            m,
            Max,
            sum(V[i]*x[i] for i in 1:3)
        )

    @constraint(m, dualcon, sum(x[i] for i in 1:3) == 1.0)

    optimize!(m)

    λ = dual(dualcon)

    ϕ_lower = objective_value(m)
    L = 5.0
    ϕ_u = λ*L

    ncols = 4
    computed_v = 3.0
    push!(V, computed_v)

    add_column_to_master!(m, x, computed_v, dualcon, ncols)
    ncols+=1
    add_column_to_master!(m, x, computed_v, dualcon, ncols)
    # #for pushing
    # push!(x, @variable(m, base_name = "x[" * string(value(ncols)) * "]", lower_bound = 0.0))

    print(JuMP.value.(x))
end

function add_column_to_master!(mlp, x, v, dualcon, ncols)
    print(typeof(x))
    print(x)
    push!(x, @variable(mlp, base_name = "x[" * string(ncols) * "]", lower_bound = 0.0))
    set_objective_coefficient(
        mlp,
        x[ncols],
        v,
        )
    set_normalized_coefficient(dualcon, x[ncols], 1)
end

dostuff()
# set_objective_coefficient(
#             m,
#             x[ncols],
#             V[ncols],
#         )

# set_normalized_coefficient(dualcon, x[ncols], 1)
