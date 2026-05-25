using RandomHAL
using StatsBase

n = parse(Int, ARGS[1])
i = ARGS[2]

function make_comparison(n, d)

    # Parameters for LASSO fitting
    kwargs = (standardize = false, nlambda = 100, folds = 10,)

    # Parameter to control how many basis functions are sampled in RandomHAL
    m = Int(round(0.5 * n * log(n)))
    sec = [[d+1]]

    # Parameter to control minimum number of nonzero entries
    minnonzero = Int(floor(sqrt(n)))

    # Functionality to sample interactions with decaying probability
    int_order = Int(round(0.5 * log(n)))
    int_weight = Weights(reverse(2 .^ (0:int_order)))

    return([
        "RandomHAL0" => (
        RandomHALRegressor(0, m, NamedTuple(), kwargs),
        RandomHALBinaryClassifier(0, m, NamedTuple(), kwargs)
        ),
        "RandomHAL1" => (
        RandomHALRegressor(1, m, NamedTuple(), kwargs),
        RandomHALBinaryClassifier(1, m, NamedTuple(), kwargs)
        ),
        "RandomHAL_keeptreat0" => (
        RandomHALRegressor(0, m, (guaranteed_sections = sec,), kwargs),
        RandomHALBinaryClassifier(0, m, NamedTuple(), kwargs)
        ),
        "RandomHAL_keeptreat1" => (
        RandomHALRegressor(1, m, (guaranteed_sections = sec,), kwargs),
        RandomHALBinaryClassifier(1, m, NamedTuple(), kwargs)
        ),
        "RandomHAL_intdecay0" => (
        RandomHALRegressor(0, m, (guaranteed_sections = sec, interaction_order_weights = int_weight), kwargs),
        RandomHALBinaryClassifier(0, m, (interaction_order_weights = int_weight,), kwargs)
        ),
        "RandomHAL_intdecay1" => (
        RandomHALRegressor(1, m, (guaranteed_sections = sec, interaction_order_weights = int_weight), kwargs),
        RandomHALBinaryClassifier(1, m, (interaction_order_weights = int_weight,), kwargs)
        ),
        "HAL0" => (
        HALRegressor(0, 0, kwargs),
        HALBinaryClassifier(0, 0, kwargs)
        ),
        "HAL1" => (
        HALRegressor(1, 0, kwargs),
        HALBinaryClassifier(1, 0, kwargs)
        ),
        "HAL_minnonzero0" => (
        HALRegressor(0, minnonzero, kwargs),
        HALBinaryClassifier(0, minnonzero, kwargs)
        ),
        "HAL_minnonzero1" => (
        HALRegressor(1, minnonzero, kwargs),
        HALBinaryClassifier(1, minnonzero, kwargs)
        )
    ])
end

# Define the SCM for this DGP
d = 5
scm, cate = binary_scm(d, d)

# Save results to directory with 
dir = basename(@__FILE__)[1:(end-3)]

# Run the simulation
simulate_binom(scm, cate, n, make_comparison(n, d), dir, i)