module TestVector
include("preamble.jl")

@testset "basic interface" begin
    xs0 = sprandn(100, 0.3)
    xs = SparseXXVector(xs0)

    @test xs[1] isa Float64
    @test xs == xs0
    @test sprint(show, xs) isa String
    @test sprint(show, "text/plain", xs) isa String
end

@testset begin
    xs0 = sprandn(100, 0.3)
    xs = SparseXXVector(xs0)
    ys = randn(100)

    @test xs0 ⋅ ys ≈ xs ⋅ ys
    @test ys ⋅ xs0 ≈ ys ⋅ xs
end

end  # module
