using RandomHAL
using StatsBase

n = parse(Int, ARGS[1])
i = ARGS[2]

function make_comparison(n, d)

    # Parameters for LASSO fitting
    kwargs0 = (standardize = false, nlambda = 100, nfolds = 10,)
    kwargs1 = (standardize = false, nlambda = 100, nfolds = 10,)

    # Parameter to control how many basis functions are sampled in RandomHAL
    m = Int(round(0.5 * n * log(n)))
    sec = [[d+2]]


    # Functionality to sample interactions with decaying probability
    int_order = Int(round(0.5 * log(n)))
    int_weight = Weights(reverse(2 .^ (0:int_order)))

    return([
        "RandomHAL0" => (
        RandomHALRegressor(0, m, NamedTuple(), kwargs0),
        RandomHALBinaryClassifier(0, m, NamedTuple(), kwargs0)
        ),
        "RandomHAL1" => (
        RandomHALRegressor(1, m, NamedTuple(), kwargs1),
        RandomHALBinaryClassifier(1, m, NamedTuple(), kwargs1)
        ),
        "RandomHAL_keeptreat0" => (
        RandomHALRegressor(0, m, (guaranteed_sections = sec,), kwargs0),
        RandomHALBinaryClassifier(0, m, NamedTuple(), kwargs0)
        ),
        "RandomHAL_keeptreat1" => (
        RandomHALRegressor(1, m, (guaranteed_sections = sec,), kwargs1),
        RandomHALBinaryClassifier(1, m, NamedTuple(), kwargs1)
        ),
        "RandomHAL_intdecay0" => (
        RandomHALRegressor(0, m, (guaranteed_sections = sec, interaction_order_weights = int_weight), kwargs0),
        RandomHALBinaryClassifier(0, m, (interaction_order_weights = int_weight,), kwargs0)
        ),
        "RandomHAL_intdecay1" => (
        RandomHALRegressor(1, m, (guaranteed_sections = sec, interaction_order_weights = int_weight), kwargs1),
        RandomHALBinaryClassifier(1, m, (interaction_order_weights = int_weight,), kwargs1)
        ),
        "HAL0" => (
        HALRegressor(0, 0, kwargs0),
        HALBinaryClassifier(0, 0, kwargs0)
        ),
        "HAL1" => (
        HALRegressor(1, 0, kwargs1),
        HALBinaryClassifier(1, 0, kwargs1)
        )
    ])
end

# Define the SCM for this DGP
d = 4
d_first = 2
scm, cate = binary_scm(d, d_first)
grid_size = 101

# Save results to directory with 
dir = basename(@__FILE__)[1:(end-3)]

# Run the simulation
simulate_binom(scm, cate, n, make_comparison(n, d), dir, i, grid_size)
