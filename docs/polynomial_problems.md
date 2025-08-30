# Polynomial Eigenvalue Problems {#polynomial-problems}


Solve P(λ) x = 0 where P(λ) = A0 + λ A1 + λ² A2 + ...

```julia
using FeastKit
coeffs = [A0, A1, A2]  # Complex or real matrices
center, radius = 0.0 + 0.0im, 1.0
res = feast_polynomial(coeffs, center, radius, M0=20)
```

Tip: Scale/shift to keep eigenvalues inside a reasonably sized circle.
