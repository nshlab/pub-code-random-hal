using DrWatson
@quickactivate :RandomHALsims
using Random

# Get the SLURM job id
id = parse(Int, ARGS[1])
#idmin = parse(Int, ARGS[2])
#idmax = parse(Int, ARGS[3])

arg_dicts = Dict(
    :i => collect(1:400),
    :n => [100, 400, 900, 1600, 2500],
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

# Set a different but deterministic seed for each job
Random.seed!(10000 + id)

# Run the appropriate script
include(dgp)

