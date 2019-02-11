# SparseXX.jl: Sparse arrays with eXperimental eXtensions

[![Build Status](https://travis-ci.com/tkf/SparseXX.jl.svg?branch=master)](https://travis-ci.com/tkf/SparseXX.jl)
[![Codecov](https://codecov.io/gh/tkf/SparseXX.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/tkf/SparseXX.jl)
[![Coveralls](https://coveralls.io/repos/github/tkf/SparseXX.jl/badge.svg?branch=master)](https://coveralls.io/github/tkf/SparseXX.jl?branch=master)

Features:

* `dot` and `mul!` implementation using [SIMD.jl].  For supported
  types, it can have up to [2x speedup].
* Arbitrary vector types can be used to represent non-zero values and
  indices (see [JuliaLang/julia#30173]).  This allows a better
  composition with [MappedArrays.jl], [LazyArrays.jl],
  [FillArrays.jl], etc.
* Optimized fused addition and multiplication:

    * `Y = Y β + D S' X`
    * `Y = Y β + S D X`
    * `Y = Y β + D₁ S₁' X₁ + ... + Dₙ Sₙ' Xₙ` [*1]
    * `Y = Y β + (D₁ S₁' + ... + Dₙ Sₙ') X` [*1]

  where:

    * `X`: matrix or a vector
    * `S`: sparse matrix (or a vector for right-most argument)
    * `D`: `Diagonal`, `UniformScaling`, or a `Number`

[*1]: when sparse matrices share the sparse structure

[SIMD.jl]: https://github.com/eschnett/SIMD.jl
[2x speedup]: https://github.com/eschnett/SIMD.jl/pull/37#issuecomment-443972203
[MappedArrays.jl]: https://github.com/JuliaArrays/MappedArrays.jl
[LazyArrays.jl]: https://github.com/JuliaArrays/LazyArrays.jl
[FillArrays.jl]: https://github.com/JuliaArrays/FillArrays.jl
[JuliaLang/julia#30173]: https://github.com/JuliaLang/julia/pull/30173
