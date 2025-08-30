# FeastKit Custom Contour Integration Examples
# Demonstrates advanced contour integration features following the original Fortran implementation

using FeastKit
using LinearAlgebra, Plots

"""
Example 1: Different Integration Methods
Compare Gauss-Legendre, Trapezoidal, and Zolotarev integration for the same problem
"""
function example_integration_methods()
    println("=== Example 1: Integration Method Comparison ===")
    
    # Define search interval
    Emin, Emax = -1.0, 1.0
    ne = 8  # Number of integration points
    
    # Generate contours with different integration methods
    contour_gauss = feast_contour_expert(Emin, Emax, ne, 0, 100)  # Gauss-Legendre
    contour_trap = feast_contour_expert(Emin, Emax, ne, 1, 100)   # Trapezoidal  
    contour_zolo = feast_contour_expert(Emin, Emax, ne, 2, 100)   # Zolotarev
    
    println("Gauss-Legendre nodes: ", contour_gauss.Zne)
    println("Trapezoidal nodes: ", contour_trap.Zne) 
    println("Zolotarev nodes: ", contour_zolo.Zne)
    
    return contour_gauss, contour_trap, contour_zolo
end

"""
Example 2: Elliptical Contours with Different Aspect Ratios
Demonstrates control over ellipse shape using the ellipse_ratio parameter
"""
function example_elliptical_contours()
    println("\\n=== Example 2: Elliptical Contour Shapes ===")
    
    Emin, Emax = -2.0, 2.0
    ne = 12
    
    # Different ellipse aspect ratios (a/b * 100)
    circle = feast_contour_expert(Emin, Emax, ne, 0, 100)     # Circle (a/b = 1.0)
    ellipse1 = feast_contour_expert(Emin, Emax, ne, 0, 50)    # a/b = 0.5 (flatter)
    ellipse2 = feast_contour_expert(Emin, Emax, ne, 0, 200)   # a/b = 2.0 (taller)
    
    println("Circle nodes (aspect=1.0): ")
    for i in 1:4  # Show first few points
        println("  Z[$i] = $(circle.Zne[i])")
    end
    
    println("Flat ellipse nodes (aspect=0.5): ")
    for i in 1:4
        println("  Z[$i] = $(ellipse1.Zne[i])")
    end
    
    println("Tall ellipse nodes (aspect=2.0): ")
    for i in 1:4 
        println("  Z[$i] = $(ellipse2.Zne[i])")
    end
    
    return circle, ellipse1, ellipse2
end

"""
Example 3: Custom Contour with User-Defined Nodes and Weights  
Shows how to use completely custom integration points
"""
function example_custom_contour()
    println("\\n=== Example 3: Custom User-Defined Contour ===")
    
    # Define custom integration nodes (e.g., rectangular contour)
    Zne = [
        -1.0 + 0.5im,   # Top left
        1.0 + 0.5im,    # Top right  
        1.0 - 0.5im,    # Bottom right
        -1.0 - 0.5im    # Bottom left
    ]
    
    # Corresponding weights for rectangular contour (trapezoidal rule)
    Wne = [
        2.0 + 0.0im,    # Horizontal segments  
        0.0 - 1.0im,    # Vertical segments
        -2.0 + 0.0im,   # Horizontal segments
        0.0 + 1.0im     # Vertical segments  
    ]
    
    # Create contour with custom nodes and weights
    custom_contour = feast_contour_custom_weights!(Zne, Wne)
    
    println("Custom rectangular contour:")
    for i in 1:length(Zne)
        println("  Node $i: Z = $(custom_contour.Zne[i]), W = $(custom_contour.Wne[i])")
    end
    
    return custom_contour
end

"""
Example 4: Rational Function Evaluation
Demonstrates evaluation of the Feast rational function using custom contours
"""
function example_rational_function()
    println("\\n=== Example 4: Rational Function Evaluation ===")
    
    # Test eigenvalues  
    lambda = [-1.5, -0.5, 0.0, 0.5, 1.5, 2.5]  # Some inside, some outside [-1,1]
    
    # Standard elliptical contour
    Emin, Emax = -1.0, 1.0
    ne = 16
    contour = feast_contour_expert(Emin, Emax, ne, 0, 100)
    
    # Evaluate rational function
    f_values = feast_rational_expert(contour.Zne, contour.Wne, lambda)
    
    println("Rational function values (should be ~1 inside contour, ~0 outside):")
    for i in 1:length(lambda)
        inside = (lambda[i] >= Emin) && (lambda[i] <= Emax)
        println("  λ = $(lambda[i]): f(λ) = $(f_values[i]) ($(inside ? "inside" : "outside"))")
    end
    
    return f_values
end

"""
Example 5: Eigenvalue Problem with Custom Contour Integration
Solve a simple eigenvalue problem using the enhanced contour integration
"""
function example_eigenvalue_problem()
    println("\\n=== Example 5: Eigenvalue Problem with Custom Contour ===")
    
    # Create a simple test matrix
    n = 10
    A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))  # Tridiagonal matrix
    B = I  # Identity matrix
    
    println("Matrix A ($(n)x$(n) tridiagonal):")
    println("  Diagonal: $(diag(A)[1:3])... ")
    println("  Off-diagonal: $(diag(A, 1)[1:3])...")
    
    # True eigenvalues for comparison (analytic solution)
    true_eigenvalues = [2 - 2*cos(k*π/(n+1)) for k in 1:n]
    println("True eigenvalues: $(true_eigenvalues[1:5])...")
    
    # Search for eigenvalues in middle range
    Emin, Emax = 1.0, 3.0
    M0 = 6  # Number of eigenvalues to find
    
    # Use different contour types 
    println("\\nUsing Gauss-Legendre integration (8 points):")
    # This would typically be used in a full Feast solver
    contour_gauss = feast_contour_expert(Emin, Emax, 8, 0, 100)
    println("Integration nodes generated: $(length(contour_gauss.Zne))")
    
    println("\\nUsing Zolotarev integration (8 points):")  
    contour_zolo = feast_contour_expert(Emin, Emax, 8, 2, 100)
    println("Integration nodes generated: $(length(contour_zolo.Zne))")
    
    return A, true_eigenvalues, contour_gauss, contour_zolo
end

"""
Run all examples
"""
function run_all_examples()
    println("FeastKit Custom Contour Integration Examples")
    println("========================================")
    
    try
        example_integration_methods()
        example_elliptical_contours()  
        example_custom_contour()
        example_rational_function()
        example_eigenvalue_problem()
        
        println("\\nAll examples completed successfully!")
        
    catch e
        println("Error running examples: $e")
        rethrow(e)
    end
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end
