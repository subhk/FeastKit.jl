module FeastKitFastGaussQuadratureExt

using FeastKit
import FastGaussQuadrature: gausslegendre

function FeastKit.gauss_legendre_point(n::Int, k::Int)
    x, w = gausslegendre(n)
    return x[k], w[k]
end

end

