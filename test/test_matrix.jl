module TestMatrix
include("preamble.jl")

@testset begin
    m = 10
    n = 3
    A0 = sprandn(m, m, 0.3)
    A = SparseXXMatrixCSC(A0)
    B = randn(m, n)

    C0 = mul!(zeros(m, n), A0, B)
    C = mul!(zeros(m, n), A, B)
    @test C ≈ C0
    C = fmul!(zeros(m, n), A, 1, B)
    @test C ≈ C0
    C = fmul!(zeros(m, n), A, 3I, B)
    @test C ≈ 3C0
    D = Diagonal(randn(m))
    C = fmul!(zeros(m, n), A, D, B)
    @test C ≈ A0 * D * B

    C0 = mul!(zeros(m, n), A0', B)
    C = mul!(zeros(m, n), A', B)
    @test C ≈ C0
    C = fmul!(zeros(m, n), 1, A', B)
    @test C ≈ C0
    C = fmul!(zeros(m, n), 3I, A', B)
    @test C ≈ 3C0
    D = Diagonal(randn(m))
    C = fmul!(zeros(m, n), D, A', B)
    @test C ≈ D * C0
end

end  # module
