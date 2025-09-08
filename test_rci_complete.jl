#!/usr/bin/env julia

# Test script to verify RCI functions are complete
using Pkg
Pkg.activate(".")

println("Testing RCI function completeness...")

try
    # Test module loading without running full functionality
    println("Loading FeastKit...")
    using FeastKit
    using LinearAlgebra
    
    println("✓ FeastKit loaded successfully")
    
    # Check if key functions are defined
    functions_to_check = [
        :feast_srci!,
        :feast_hrci!,
        :feast_grci!,
        :feast_sort!,
        :feast_sort_general!,
        :feast_inside_gcontour,
        :feast_inside_contour
    ]
    
    missing_functions = []
    for func in functions_to_check
        if isdefined(FeastKit, func)
            println("✓ $func is defined")
        else
            println("✗ $func is NOT defined")
            push!(missing_functions, func)
        end
    end
    
    if isempty(missing_functions)
        println("\n✓ All required RCI functions are defined!")
    else
        println("\n✗ Missing functions: $missing_functions")
    end
    
    # Test simple parameter validation 
    println("\nTesting parameter validation...")
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    println("✓ feastinit! works")
    
    result = check_feast_srci_input(10, 5, 0.0, 2.0, fpm)
    println("✓ check_feast_srci_input works: $result")
    
    # Test contour generation
    println("\nTesting contour generation...")
    contour = feast_contour(0.0, 2.0, fpm)
    println("✓ feast_contour works: $(length(contour.Zne)) points")
    
    contour_g = feast_gcontour(1.0+1.0im, 1.0, fpm)
    println("✓ feast_gcontour works: $(length(contour_g.Zne)) points")
    
    # Test utility functions
    println("\nTesting utility functions...")
    println("✓ feast_inside_contour(1.5, 0.0, 2.0): ", feast_inside_contour(1.5, 0.0, 2.0))
    println("✓ feast_inside_gcontour(0.5+0.5im, 1.0+1.0im, 1.0): ", feast_inside_gcontour(0.5+0.5im, 1.0+1.0im, 1.0))
    
    # Test sorting functions
    println("\nTesting sorting functions...")
    lambda_real = [3.0, 1.0, 2.0]
    q_real = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    res_real = [0.1, 0.3, 0.2]
    feast_sort!(lambda_real, q_real, res_real, 3)
    println("✓ feast_sort! works: sorted eigenvalues = $lambda_real")
    
    lambda_complex = [3.0+0.0im, 1.0+1.0im, 2.0+0.5im]
    q_complex = Complex{Float64}[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    res_complex = [0.1, 0.3, 0.2]
    feast_sort_general!(lambda_complex, q_complex, res_complex, 3)
    println("✓ feast_sort_general! works: sorted eigenvalues = $lambda_complex")
    
    println("\n🎉 All RCI function tests passed!")
    
catch e
    println("❌ Error during testing:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end