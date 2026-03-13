using DrWatson
@quickactivate "RandomHALsims"

make_models(n, k) = [
    "RandomHAL0" => (
    RandomHALRegressor(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "RandomHAL1" => (
    RandomHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    )
]

scm, cate = binary_scm(40, 20, 0.1)
result = [simulate_binom(scm, cate, n, 100, make_models(n, 4)) for n in [100, 400, 900, 1600]]