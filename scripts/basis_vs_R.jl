using RCall
using RandomHAL
using Test
import RandomHAL: hal_basis
using StatsBase

#reval("install.packages('hal9001')")
reval("library('hal9001')")

@testset "Compare basis between Julia and R versions" begin
    d = 6
    n = 20
    x = R"matrix(c(rnorm($n * $(d-1)),rbinom(100, 1, 0.5)),$n,$d)"
    basis = R"enumerate_basis($x)"
    Rmat = rcopy(R"as.matrix(make_design_matrix($x, $basis))")[:, 2:end] # remove intercept
    Jmat = RandomHAL.highly_adaptive_basis(rcopy(x),(Float64, Float64, Float64, Float64, Float64, Bool), 0)

    # Test whether the matrices are the same size
    @test size(Rmat) == size(Jmat)

    # Check whether all of the columns are the same
    cols = [c for c in eachcol(Jmat)]
    @test mean([c in cols for c in eachcol(Rmat)]) == 1.0
end
