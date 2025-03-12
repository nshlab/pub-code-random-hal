using DrWatson
@quickactivate
using Test

# An example of a test set
@testset "test" begin
    @test 1 + 1 == 2
    @test_throws MethodError "1" + 1 == 2
end