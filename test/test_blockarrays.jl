module TestBlockArrays
include("preamble.jl")

using BlockArrays
using FillArrays

allocblockarrays(;
        rowblocksizes = [4, 6],
        colblocksizes = [3, 7],
        blocksizes = (rowblocksizes, colblocksizes),
        sprototype = spzeros(1, 1)',
        stype = typeof(sprototype),
        ) =
    BlockArray(undef_blocks, stype, blocksizes...)

function randomblocksparse(; p = 0.3, kwargs...)
    S = allocblockarrays(; kwargs...)

    for i in 1:nblocks(S, 1),
        j in 1:nblocks(S, 2)

        m = blocksize(S, 1, i)
        n = blocksize(S, 2, j)
        S.blocks[i, j] = sprand(n, m, p)'
    end

    return S
end

chained_test_params = let params = []
    n = 5

    S1 = randomblocksparse()
    S2 = spshared(S1); foreach(randn! ∘ nonzeros ∘ parent, S2.blocks)
    S3 = spshared(S1); foreach(randn! ∘ nonzeros ∘ parent, S3.blocks)
    D1 = true
    D2 = Diagonal(randn(size(S1, 1)))
    D3 = Eye(size(S1, 1))
    X1 = randn(size(S1, 2), n)
    X2 = randn(size(S1, 2), n)
    X3 = randn(size(S1, 2), n)

    desired = D1 * S1 * X1
    push!(params, (
        label = "#DS=1",
        B = asfmulable((D1, S1)),
        X = X1,
        Y = zero(desired),
        desired = desired,
    ))
    desired = (D1 * S1 + D2 * S2) * X1
    push!(params, (
        label = "#DS=2",
        B = asfmulable((D1, S1), (D2, S2)),
        X = X1,
        Y = zero(desired),
        desired = desired,
    ))
    desired = (D1 * S1 + D2 * S2 + D3 * S3) * X1
    push!(params, (
        label = "#DS=3",
        B = asfmulable((D1, S1), (D2, S2), (D3, S3)),
        X = X1,
        Y = zero(desired),
        desired = desired,
    ))

    desired = (D1 * S1 * X1,)
    push!(params, (
        label = "#DSX=1",
        B = asfmulable((D1, S1)),
        X = (X1,),
        Y = zero.(desired),
        desired = desired,
    ))
    desired = (D1 * S1 * X1, D2 * S2 * X2)
    push!(params, (
        label = "#DSX=2",
        B = asfmulable((D1, S1), (D2, S2)),
        X = (X1, X2),
        Y = zero.(desired),
        desired = desired,
    ))
    desired = (D1 * S1 * X1, D2 * S2 * X2, D3 * S3 * X3)
    push!(params, (
        label = "#DSX=3",
        B = asfmulable((D1, S1), (D2, S2), (D3, S3)),
        X = (X1, X2, X3),
        Y = zero.(desired),
        desired = desired,
    ))

    params
end

@testset "ChainedFMulShared $(p.label)" for p in chained_test_params
    @unpack desired, Y, B, X = p

    actual = fmul!(Y, B, X)
    @test actual === Y

    if desired isa Tuple
        @test length(actual) == length(desired)
        @testset for i in 1:length(actual)
            @test actual[i] ≈ desired[i]
        end
    else
        @test actual ≈ desired
    end
end

end  # module
