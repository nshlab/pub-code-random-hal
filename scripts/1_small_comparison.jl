using DrWatson
@quickactivate :RandomHALsims

using RandomHAL

make_comparison(n, k) = [
    "RandomHAL0" => (
    RandomHALRegressor(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "RandomHAL1" => (
    RandomHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "HAL0" => (
    HALRegressor(0),
    HALBinaryClassifier(0)
    ),
    "HAL1" => (
    HALRegressor(1),
    HALBinaryClassifier(1)
    )
]

scm, cate = small_scm()
result = [simulate_binom(scm, cate, n, 5, make_comparison(n, 2)) for n in [50, 100]]