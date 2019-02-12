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

fmul_shared_test_params = let params = []
    m = 10
    n = 3

    S1 = sprandn(m, m, 0.3)
    S2 = spshared(S1)
    S3 = spshared(S1)
    randn!(nonzeros(S2))
    randn!(nonzeros(S3))
    push!(params, (
        label = "Xn :: Matrix",
        D1 = Diagonal(randn(m)),
        D2 = Diagonal(randn(m)),
        D3 = Diagonal(randn(m)),
        S1 = S1,
        S2 = S2,
        S3 = S3,
        X1 = randn(m, n),
        X2 = randn(m, n),
        X3 = randn(m, n),
    ))

    push!(params, (
        label = "Xn :: Vector",
        D1 = Diagonal(randn(m)),
        D2 = Diagonal(randn(m)),
        D3 = Diagonal(randn(m)),
        S1 = S1,
        S2 = S2,
        S3 = S3,
        X1 = randn(m),
        X2 = randn(m),
        X3 = randn(m),
    ))

    S1 = sprandn(m, m, 0.3)
    S2 = spshared(S1)
    S3 = spshared(S1)
    randn!(nonzeros(S2))
    randn!(nonzeros(S3))
    push!(params, (
        label = "mixed D type",
        D1 = randn(),
        D2 = randn() * I,
        D3 = Diagonal(randn(m)),
        S1 = S1,
        S2 = S2,
        S3 = S3,
        X1 = randn(m, n),
        X2 = randn(m, n),
        X3 = randn(m, n),
    ))

    params
end

@testset "is_shared_simd $(p.label)" for p in fmul_shared_test_params[1:2]
    @unpack D1, D2, D3, S1, S2, S3, X1, X2, X3 = p

    @test SparseXX.is_shared_simd3(((D1, S1', X1), (D2, S2', X2)))
    @test SparseXX.is_shared_simd2(((D1, S1'), (D2, S2'), X1))
    @test SparseXX.is_shared_simd3(((D1, S1', X1),
                                    (D2, S2', X2),
                                    (D3, S3', X3)))
    @test SparseXX.is_shared_simd2(((D1, S1'),
                                    (D2, S2'),
                                    (D3, S3'),
                                    X1))
end

@testset "fmul_shared! $(p.label)" for p in fmul_shared_test_params
    @unpack D1, D2, D3, S1, S2, S3, X1, X2, X3 = p

    Y = fmul_shared!(zero(X1), (D1, S1', X1), (D2, S2', X2))
    @test Y ≈ D1 * S1' * X1 + D2 * S2' * X2

    Y1, Y2 = fmul_shared!((zero(X1), zero(X1)), (D1, S1', X1), (D2, S2', X2))
    @test Y1 ≈ D1 * S1' * X1
    @test Y2 ≈ D2 * S2' * X2

    Y = fmul_shared!(zero(X1), (D1, S1'), (D2, S2'), X1)
    @test Y ≈ (D1 * S1' + D2 * S2') * X1

    Y1, Y2 = fmul_shared!((zero(X1), zero(X1)), (D1, S1'), (D2, S2'), X1)
    @test Y1 ≈ D1 * S1' * X1
    @test Y2 ≈ D2 * S2' * X1

    Y = fmul_shared!(zero(X1),
                     (D1, S1', X1),
                     (D2, S2', X2),
                     (D3, S3', X3))
    @test Y ≈ D1 * S1' * X1 + D2 * S2' * X2 + D3 * S3' * X3

    Y1, Y2, Y3 = fmul_shared!((zero(X1), zero(X1), zero(X1)),
                              (D1, S1', X1),
                              (D2, S2', X2),
                              (D3, S3', X3))
    @test Y1 ≈ D1 * S1' * X1
    @test Y2 ≈ D2 * S2' * X2
    @test Y3 ≈ D3 * S3' * X3

    Y = fmul_shared!(zero(X1),
                     (D1, S1'),
                     (D2, S2'),
                     (D3, S3'),
                     X1)
    @test Y ≈ (D1 * S1' + D2 * S2' + D3 * S3') * X1

    Y1, Y2, Y3 = fmul_shared!((zero(X1), zero(X1), zero(X1)),
                              (D1, S1'),
                              (D2, S2'),
                              (D3, S3'),
                              X1)
    @test Y1 ≈ D1 * S1' * X1
    @test Y2 ≈ D2 * S2' * X1
    @test Y3 ≈ D3 * S3' * X1
end

end  # module
