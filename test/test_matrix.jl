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

    C0 = mul!(zeros(m, n), A0', B)
    C = mul!(zeros(m, n), A', B)
    @test C ≈ C0
end

end  # module
