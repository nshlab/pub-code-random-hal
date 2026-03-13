module RandomHALsims

    using RandomHAL
    using CausalTables
    using Distributions
    using LogExpFunctions
    using MLJ
    using Plots
    using Tables
    using Copulas
    using LinearAlgebra
    using CSV
    using DataFrames
    using DataFramesMeta
    using Tables
    using DrWatson

    include("simulate.jl")
    export binary_scm, simulate_binom

end