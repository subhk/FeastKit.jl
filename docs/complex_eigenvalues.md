# Complex Eigenvalues (Non-Hermitian) {#complex_eigenvalues}

```@id complex_eigenvalues
```


For non-Hermitian problems, search in a circular region of the complex plane:

```julia
using FeastKit
A = your_nonsymmetric_matrix()
B = Matrix(I, size(A)...)  # identity
center, radius = 1.0 + 0.5im, 2.0
res = feast_general(A, B, center, radius, M0=20)
```

If you need custom contours beyond circles, see Custom Contours.
