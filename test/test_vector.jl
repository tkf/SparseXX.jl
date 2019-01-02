module TestVector
include("preamble.jl")

@testset begin
    xs0 = sprandn(100, 0.3)
    xs = SparseXXVector(xs0)
    ys = randn(100)

    @test xs0 ⋅ ys ≈ xs ⋅ ys
    @test ys ⋅ xs0 ≈ ys ⋅ xs
end

end  # module
