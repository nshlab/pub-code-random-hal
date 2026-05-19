using DrWatson
@quickactivate :RandomHALsims

# Get the SLURM job id
id = parse(Int, ARGS[1])

arg_dicts = Dict(
    :i => collect(1:100),
    :n => [100, 400, 900, 1600],
    :dgp => ["3_small_comparison.jl", "4_large_randomhal.jl"]
)

arg_list = dict_list(arg_dicts)
println("Total jobs: ", length(arg_list))
dgp, n, i = values(arg_list[id])
println("Running DGP $(dgp), n = $(n), simulation $(i) ")

# Reset the ARGS and add the appropriate ones
# to mimic the behavior of adding script arguments and running the DGP directly
empty!(ARGS)
append!(ARGS, [string(n), string(i)])

# Run the appropriate script
include(dgp)

