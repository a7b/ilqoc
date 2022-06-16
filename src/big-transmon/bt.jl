WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "big-transmon", "systemc.jl"))

using TrajectoryOptimization
const TO = TrajectoryOptimization
using Altro
using RobotDynamics
const RD = RobotDynamics
#paths
const EXPERIMENT_META = "big"
const EXPERIMENT_NAME = "transmon"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

#const SA_C64 = SA{ComplexF64}
Rx(θ) = [cos(θ/2) -1im * sin(θ/2);
        -1im * sin(θ/2) cos(θ/2)]

function run_traj(;evolution_time = 50.0, solver_type = altro, dt = 0.1, ω = 0.0, α = -2π * 0.143, levels = 10, a_max = 2π * 100e-3, sqrtbp = false, qs = [1e2, 1e0, 1e0, 1e-1] ,smoke_test=false,
    projected_newton = false, constraint_tol=1e-8, al_tol=1e-4,pn_steps=2, max_penalty=1e11, verbose=true, save=true, max_iterations=Int64(2e5),
    nf = false, nf_tol = 0., max_cost_value=1e8, benchmark=false, al_iters = 30)

    model = Transmon(ω, α, levels)
    dmodel = RD.DiscretizedDynamics{RD.RK3}(model)
    n = RD.state_dim(model)
    m = RD.control_dim(model)
    #initial states
    initial_states = [
                    [1; zeros(levels - 1)], #|g>
                    [0.; 1.; zeros(levels-2)], #|e>
                    [1/sqrt(2); 1im/sqrt(2); zeros(levels - 2)], #|g> + i|e>
                    [1/sqrt(2); -1/sqrt(2); zeros(levels - 2)] #|g> - |e>
                    ]

    final_states = [append!(Rx(π)*initial_states[j][1:2], zeros(levels - 2)) for j in 1:length(initial_states)]


    x0 = zeros(n)
    for i = 1:nstates(model)
        setqstate!(model, x0, initial_states[i], i)
    end

    xf = zeros(n)
    for i = 1:nstates(model)
        setqstate!(model, xf, final_states[i], i)
    end

    t0 = 0.
    N = Int(floor(evolution_time/dt)) + 1
    ts = zeros(N)
    ts[1] = t0
    for k = 1:N - 1
        ts[k+1] = ts[k] + dt
    end
    hdim_iso_states = nstates(model)*2*model.levels
    ncontrol = RD.control_dim(model)

    Q = Diagonal([
        fill(qs[1], hdim_iso_states); # ψ1, ψ2, ψ3, ψ4
        fill(qs[2], ncontrol); # a
        fill(qs[3], ncontrol); # ∂a
    ])
    Qf = Q * N
    R = Diagonal(fill(qs[4], ncontrol)) #∂2a
    obj = LQRObjective(Q, R, Qf, xf, N)

    #initial guesses

    U0 = [SVector{m}(
        fill(a_max/100, ncontrol)
    ) for k = 1:N-1]
    X0 = [copy(x0) for k = 1:N]

    #constraints
    cons = ConstraintList(n, m, N)
    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[control_idx(model)] .= a_max
    x_min = fill(-Inf, n)
    x_min[control_idx(model)] .= -a_max

    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[control_idx(model)] .= 0
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[control_idx(model)] .= 0

    control_bound = BoundConstraint(n, m, x_max = x_max, x_min = x_min)
    add_constraint!(cons, control_bound, 1:N-2)

    control_bound_bndry = BoundConstraint(n, m, x_max = x_max_boundary, x_min = x_min_boundary)
    add_constraint!(cons, control_bound_bndry, N-1)

    goal = GoalConstraint(xf, 1:hdim_iso_states)
    add_constraint!(cons, goal, N)

    # Norm Constraint
    norm_cons = map(1:nstates(model)) do i
        NormConstraint(n, m, 1.0, TO.Equality(), getqstateinds(model, i))
    end
    for con in norm_cons
        add_constraint!(cons, con, 1:N)
    end

    prob = Problem(dmodel, obj, x0, evolution_time, xf=xf, X0=X0, U0=U0, constraints = cons)
    verbose_pn = verbose ? true : false
    #verbose_ = verbose ? 2 : 0
    constraint_tolerance = solver_type == altro ? constraint_tol : al_tol
    iterations_inner = smoke_test ? 1 : 300
    iterations_outer = smoke_test ? 1 : al_iters
    n_steps = smoke_test ? 1 : pn_steps

    solver = ALTROSolver(prob, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose = 4,
                 projected_newton=projected_newton, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations,
                 max_cost_value=max_cost_value, dynamics_funsig = RD.InPlace(), use_static=Val(false))
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

    # post-process
    acontrols_raw = TO.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(control_idx(model))
    d2cidx_arr = Array(1:control_count(model))
    #cmax = TrajectoryOptimization.max_violation(solver)
    #cmax_info = TrajectoryOptimization.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "controls_idx" => cidx_arr,
        "d2controls_dt2_idx" => d2cidx_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "hdim_iso" => model.levels*2,
        "Q" => Q_arr,
        "Qf" => Qf_arr,
        "R" => R_arr,
        "ts" => ts,
        #"cmax" => cmax,
        #"cmax_info" => cmax_info,
        "dt" => dt,
        #"derivative_count" => derivative_order,
        "solver_type" => Integer(solver_type),
        "sqrtbp" => Integer(sqrtbp),
        "max_penalty" => max_penalty,
        "constraint_tol" => constraint_tol,
        "al_tol" => al_tol,
        "save_type" => Integer(jl),
        #"integrator_type" => Integer(integrator_type),
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost_value" => max_cost_value,
        "transmon_state_count" => nstates(model)
    )


    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
        result["save_file_path"] = save_file_path
    end

    result = benchmark ? benchmark_result : result
    plot_population(save_file_path)
    control_path = generate_file_path("png", "controls", SAVE_PATH)
    plot_controls([save_file_path], control_path)
    return result
end

function plot_population(save_file_path; title="", xlabel="Time (ns)", ylabel="Population",
                         legend=:bottomright)
    # grab
    save_file = read_save(save_file_path)
    transmon_state_count = save_file["transmon_state_count"]
    hdim_iso = save_file["hdim_iso"]
    ts = save_file["ts"]
    astates = save_file["astates"]
    N = size(astates, 1)
    d = Int(hdim_iso/2)
    state1_idx = Array(1:hdim_iso)
    state2_idx = Array(1 + hdim_iso: hdim_iso + hdim_iso)
    state3_idx = Array(1 + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso)
    state4_idx = Array(1 + hdim_iso + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso + hdim_iso)
    # make labels
    transmon_labels = ["g", "e", "f", "h"][1:transmon_state_count]

    # plot
    fig = Plots.plot(dpi=DPI, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    pops = zeros(N, d)
    pops2 = zeros(N, d)
    pops3 = zeros(N, d)
    pops4 = zeros(N, d)
    for k = 1:N
        ψ = get_vec_uniso(astates[k, state1_idx])
        ψ2 = get_vec_uniso(astates[k, state2_idx])
        ψ3 = get_vec_uniso(astates[k, state3_idx])
        ψ4 = get_vec_uniso(astates[k, state4_idx])
        pops[k, :] = map(x -> abs(x)^2, ψ)
        pops2[k, :] = map(x -> abs(x)^2, ψ2)
        pops3[k, :] = map(x -> abs(x)^2, ψ3)
        pops4[k, :] = map(x -> abs(x)^2, ψ4)
    end
    styles = [:solid, :dash, :dot, :dashdot]
    for i = 1:4
        label = transmon_labels[i]
        style = styles[i]
        Plots.plot!(ts, pops[:, i], linestyle = style, lc = "cornflowerblue"  , label=label)
        Plots.plot!(ts, pops2[:, i], linestyle = style, lc = "darkorange", label = label)
        Plots.plot!(ts, pops3[:, i], linestyle = style, lc = "darkseagreen", label = label)
        Plots.plot!(ts, pops4[:, i], linestyle = style, lc = "violet", label = label)
    end
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end
