using DrWatson
@quickactivate :RandomHALsims

using RandomHAL

function make_comparison(n, d)

    # Parameters for LASSO fitting
    nλ = 100
    folds = 10

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
        RandomHALRegressor(0, nλ, folds, m, 0, NamedTuple()),
        RandomHALBinaryClassifier(0, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL1" => (
        RandomHALRegressor(1, nλ, folds, m, 0, NamedTuple()),
        RandomHALBinaryClassifier(1, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL_minnonzero0" => (
        RandomHALRegressor(0, nλ, folds, m, minnonzero, NamedTuple()),
        RandomHALBinaryClassifier(0, nλ, folds, m, minnonzero, NamedTuple())
        ),
        "RandomHAL_minnonzero1" => (
        RandomHALRegressor(1, nλ, folds, m, 0, NamedTuple()),
        RandomHALBinaryClassifier(1, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL_keeptreat0" => (
        RandomHALRegressor(0, nλ, folds, m, 0, (guaranteed_sections = sec,)),
        RandomHALBinaryClassifier(0, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL_keeptreat1" => (
        RandomHALRegressor(1, nλ, folds, m, 0, (guaranteed_sections = sec,)),
        RandomHALBinaryClassifier(1, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL_intdecay0" => (
        RandomHALRegressor(0, nλ, folds, m, 0, (guaranteed_sections = sec, interaction_order_weights = int_weight)),
        RandomHALBinaryClassifier(0, nλ, folds, m, 0, NamedTuple())
        ),
        "RandomHAL_intdecay1" => (
        RandomHALRegressor(1, nλ, folds, m, 0, (guaranteed_sections = sec, interaction_order_weights = int_weight)),
        RandomHALBinaryClassifier(1, nλ, folds, m, 0, NamedTuple())
        )
    ])
end

d = 40
scm, cate = binary_scm(d, 8)
result = [simulate_binom(scm, cate, n, 100, make_comparison(n, d)) for n in [100, 400, 900, 1600]] 