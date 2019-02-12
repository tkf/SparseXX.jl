struct ChainedFMulShared{
        N,
        TS <: NTuple{N,Tuple{Vararg{Tuple{DiagonalLike,AbstractMatrix}}}},
        YR <: NTuple{N,AbstractRange},
        XR <: NTuple{N,AbstractRange}}
    terms::TS
    yranges::YR
    xranges::XR
end

ChainedFMulShared(terms::TS,
                  yranges::YR,
                  xranges::XR) where {N,
                                      TS <: NTuple{N},
                                      YR <: NTuple{N},
                                      XR <: NTuple{N}} =
    ChainedFMulShared{N,TS,YR,XR}(terms, yranges, xranges)

@inline aschainedargs(Y, B, X) =
    _aschainedargs(B.terms, B.yranges, B.xranges, Y, X)

@inline _aschainedargs(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Any, ::Any) = ()
@inline _aschainedargs(terms, yranges, xranges, Y, X) =
    ((_getlhs(_vrows(Y, yranges[1])),                # (Y, β)
      _getrhs(terms[1], _vrows(X, xranges[1]))...),  # rhs...
     _aschainedargs(tail(terms), tail(yranges), tail(xranges), Y, X)...)

@inline _vrows(M::AbstractMatrix, i) = view(M, i, :)
@inline _vrows(v::AbstractVector, i) = view(M, i)
@inline _vrows(Xs::Tuple, i) = _vrows.(Xs, Ref(i))

@inline _getrhs(DSs::Tuple, X::AbstractArray) = (DSs..., X)
@inline _getrhs(::Tuple{}, ::Tuple{}) = ()
@inline _getrhs(DSs::Tuple, Xs::Tuple) = ((DSs[1]..., Xs[1]),
                                          _getrhs(tail(DSs), tail(Xs))...)

@inline _getlhs(Y) = (Y, true)  # `β=true` to accumulate all terms/blocks
@inline _getlhs(Y::Tuple) = _getlhs.(Y)

_foreachsplat(f, ::Tuple{}) = nothing
_foreachsplat(f, xs::Tuple) = (f(xs[1]...); _foreachsplat(f, tail(xs)))

function fmul!(Y_, B::ChainedFMulShared, X)
    Y = preprocess_Yβ(canonicalize_Yβ(Y_))
    _foreachsplat(fmul_shared!, aschainedargs(Y, B, X))
    return Y
end

"""
    asfmulable(rhs...) -> fmulable

Return an object `fmulable` which can be used as `fmul!(Y, fmulable, X)`.
"""
function asfmulable end
