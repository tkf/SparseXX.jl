module TestMatrix
include("preamble.jl")

@testset "basic interface" begin
    m = 10
    n = 3
    A0 = sprandn(m, m, 0.3)
    A = SparseXXMatrixCSC(A0)
    @test A[1, 1] isa Float64
    @test A == A0
    @test sprint(show, A) isa String
    @test sprint(show, "text/plain", A) isa String
end

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

@testset "fmul_shared!" begin
    m = 10
    n = 3
    D1 = Diagonal(randn(m))
    D2 = Diagonal(randn(m))
    D3 = Diagonal(randn(m))
    S1 = sprandn(m, m, 0.3)
    S2 = spshared(S1)
    S3 = spshared(S1)
    randn!(nonzeros(S2))
    randn!(nonzeros(S3))
    X1 = randn(m, n)
    X2 = randn(m, n)
    X3 = randn(m, n)

    @test SparseXX.is_shared_csr_simd(((D1, S1', X1), (D2, S2', X2)))
    Y = fmul_shared!(zero(X1), (D1, S1', X1), (D2, S2', X2))
    @test Y ≈ D1 * S1' * X1 + D2 * S2' * X2

    @test SparseXX.is_shared_csr_simd(((D1, S1', X1),
                                       (D2, S2', X2),
                                       (D3, S3', X3)))
    Y = fmul_shared!(zero(X1),
                     (D1, S1', X1),
                     (D2, S2', X2),
                     (D3, S3', X3))
    @test Y ≈ D1 * S1' * X1 + D2 * S2' * X2 + D3 * S3' * X3
end

end  # module
