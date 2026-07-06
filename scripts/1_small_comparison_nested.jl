using DrWatson
@quickactivate :RandomHALsims

using RandomHAL

make_comparison(n, k) = [
    "RandomHAL0" => (
    NRHALRegressor(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    NRHALClassifier(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "RandomHAL1" => (
    NRHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    NRHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
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

scm, cate = binary_scm(4, 4)
result = [simulate_binom(scm, cate, n, 100, make_comparison(n, 2)) for n in [100, 400, 900, 1600]]