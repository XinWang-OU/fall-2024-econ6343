# Question 1
# Import necessary packages
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Statistics, Random, ForwardDiff

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Set up X (individual-specific covariates) and Z (alternative-specific covariates)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Function to compute the multinomial logit probabilities
function logit_probabilities(X, Z, β, γ)
    N, J = size(Z)
    utilities = X * β .+ γ * (Z .- Z[:, J])  # Alternative-specific utilities
    exp_utilities = exp.(utilities)
    denom = 1 .+ sum(exp_utilities[:, 1:J-1], dims=2)
    probabilities = hcat(exp_utilities[:, 1:J-1] ./ denom, 1 ./ denom)
    return probabilities
end

# Likelihood function (negative log-likelihood)
function log_likelihood(params)
    β = params[1:size(X, 2)]
    γ = params[end]
    probs = logit_probabilities(X, Z, β, γ)
    likelihood = sum(log.(probs[CartesianIndex(i, y[i])] for i in 1:length(y)))
    return -likelihood  # Negative for minimization
end

# Initial guess for parameters
β_init = randn(size(X, 2))
γ_init = randn(1)
params_init = vcat(β_init, γ_init)

# Use Optim with automatic differentiation (ForwardDiff) to minimize the log-likelihood
result = optimize(log_likelihood, params_init, BFGS(), autodiff = :forward)

# Extract the optimized parameter estimates
params_opt = Optim.minimizer(result)

# Use ForwardDiff to compute the Hessian for standard errors
hessian_approx = ForwardDiff.hessian(log_likelihood, params_opt)
standard_errors = sqrt.(diag(inv(hessian_approx)))

# Print out results
println("Estimated coefficients: ", params_opt)
println("Standard errors: ", standard_errors)


# Question 2
# The new estimate of γ in PS4 is more reasonable compared to the PS3 result. 
# This is primarily due to the use of automatic differentiation, which improves the efficiency and stability of the optimization process. 
# The smaller standard errors also indicate that the estimates are more precise.
# The result in PS4 gives us greater confidence in the impact of wage differences (as an alternative-specific covariate) on occupational choices, with a clear negative effect as expected.

#Question 3(a)

using Distributions
include("lgwt.jl") # make sure the function gets read in
# Import necessary package for distribution
using Distributions

# Define the normal distribution N(0, 1) (mean=0, std=1)
d = Normal(0, 1)

# Get quadrature nodes and weights for 7 grid points over the interval [-4, 4]
nodes, weights = lgwt(7, -4, 4)

# Now compute the integral over the density and verify it's 1
# This should approximate the integral ∫ φ(x) dx where φ(x) is the normal pdf
density_integral = sum(weights .* pdf.(d, nodes))
println("Density integral (should be close to 1): ", density_integral)

# Now compute the expectation and verify it's 0
expectation = sum(weights .* nodes .* pdf.(d, nodes))
println("Expectation (should be close to 0): ", expectation)

# For further practice:
# Define a new normal distribution N(0, 2) (mean=0, std=sqrt(2))
d2 = Normal(0, sqrt(2))

# Compute the integral ∫ x^2 φ(x) dx where φ(x) is the pdf of N(0, 2)
variance_integral_7_points = sum(weights .* (nodes.^2) .* pdf.(d2, nodes))
println("Variance integral with 7 points (should be close to 4): ", variance_integral_7_points)

# Try with 10 quadrature points instead of 7
nodes_10, weights_10 = lgwt(10, -4, 4)
variance_integral_10_points = sum(weights_10 .* (nodes_10.^2) .* pdf.(d2, nodes_10))
println("Variance integral with 10 points (should be close to 4): ", variance_integral_10_points)

# Question 3(b)
include("lgwt.jl") 

# Define the normal distribution N(0, sqrt(2))
mu = 0
sigma = sqrt(2)
d = Normal(mu, sigma)

# Perform 7-point Gauss-Legendre quadrature
nodes_7, weights_7 = lgwt(7, -5*sigma, 5*sigma)
integral_7_points = sum(weights_7 .* (nodes_7.^2) .* pdf.(d, nodes_7))
println("Integral with 7 points: ", integral_7_points)

# Perform 10-point Gauss-Legendre quadrature
nodes_10, weights_10 = lgwt(10, -5*sigma, 5*sigma)
integral_10_points = sum(weights_10 .* (nodes_10.^2) .* pdf.(d, nodes_10))
println("Integral with 10 points: ", integral_10_points)

# True variance of N(0, sqrt(2)) is sigma^2
true_variance = sigma^2
println("True variance: ", true_variance)

# Question 3(c)
using Distributions, Random

# Define the normal distribution N(0, sqrt(2))
mu = 0
sigma = sqrt(2)
d = Normal(mu, sigma)

# Monte Carlo simulation parameters
D = 1_000_000  # Number of random draws
a, b = -5*sigma, 5*sigma  # Integration bounds

# Draw D random samples from the normal distribution N(0, sqrt(2))
X = rand(d, D)

# Approximate the integrals using Monte Carlo
# (1) Integral of x^2 * f(x) over [-5σ, 5σ]
integral_x2 = (b - a) * mean(X[X .>= a .&& X .<= b].^2)

# (2) Integral of x * f(x) over [-5σ, 5σ]
integral_x = (b - a) * mean(X[X .>= a .&& X .<= b])

# (3) Integral of f(x) over [-5σ, 5σ]
integral_f = (b - a) * length(X[X .>= a .&& X .<= b]) / D

# Display the results
println("Monte Carlo integral of x^2 f(x): ", integral_x2, " (Expected: 4)")
println("Monte Carlo integral of x f(x): ", integral_x, " (Expected: 0)")
println("Monte Carlo integral of f(x): ", integral_f, " (Expected: 1)")

# Question 3(d)
# Include the lgwt.jl file to use the quadrature function
include("lgwt.jl")  # Adjust the path if necessary

# Import necessary packages
using Distributions, Random

# Define the normal distribution N(0, sqrt(2))
mu = 0
sigma = sqrt(2)
d = Normal(mu, sigma)

# Quadrature method (7-point and 10-point)
nodes_7, weights_7 = lgwt(7, -5*sigma, 5*sigma)
quadrature_7 = sum(weights_7 .* (nodes_7.^2) .* pdf.(d, nodes_7))

nodes_10, weights_10 = lgwt(10, -5*sigma, 5*sigma)
quadrature_10 = sum(weights_10 .* (nodes_10.^2) .* pdf.(d, nodes_10))

# Monte Carlo method
D = 1_000_000  # Number of random draws
a, b = -5*sigma, 5*sigma  # Integration bounds
X = rand(d, D)

# Monte Carlo approximation
monte_carlo = (b - a) * mean(X[X .>= a .&& X .<= b].^2)

# Output the results
println("Quadrature result with 7 points: ", quadrature_7)
println("Quadrature result with 10 points: ", quadrature_10)
println("Monte Carlo result: ", monte_carlo)
println("True variance (expected value): 4")

# Comparison of quadrature and Monte Carlo weights and nodes:
# Quadrature: Nodes are chosen carefully, and weights differ for each node.
println("Quadrature nodes (7-point): ", nodes_7)
println("Quadrature weights (7-point): ", weights_7)

# Monte Carlo: Nodes are random, and weights are equal for all nodes.
println("Monte Carlo nodes (sample): ", X[1:10])  # Show first 10 samples for illustration
println("Monte Carlo weight: ", (b - a) / D)  # Equal weight for all samples

# Question 4
# Include the lgwt.jl file to use Gauss-Legendre quadrature function
include("lgwt.jl")

# Import necessary packages
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, ForwardDiff, Distributions

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Set up X (individual-specific covariates) and Z (alternative-specific covariates)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Define the number of quadrature points
K = 7  # You can use more points for higher accuracy
nodes, weights = lgwt(K, -4, 4)  # Get quadrature points and weights for N(0, 1)

# Define the logit probability function with random gamma using quadrature
function logit_probabilities_with_quadrature(X, Z, β, μγ, σγ)
    N, J = size(Z)
    utilities = X * β  # Alternative-specific utilities

    # Initialize probability matrix
    probabilities = zeros(N, J)
    
    # Loop over quadrature points
    for r in 1:K
        ξ_r = μγ + σγ * nodes[r]  # Quadrature point for gamma
        exp_utilities = exp.(utilities .+ ξ_r * (Z .- Z[:, J]))  # Update utilities with random gamma
        denom = 1 .+ sum(exp_utilities[:, 1:J-1], dims=2)
        temp_probs = hcat(exp_utilities[:, 1:J-1] ./ denom, 1 ./ denom)
        
        # Sum the weighted probabilities
        probabilities += weights[r] * temp_probs
    end
    
    return probabilities
end

# Likelihood function (negative log-likelihood)
function log_likelihood_quadrature(params)
    β = params[1:size(X, 2)]  # Beta coefficients
    μγ = params[end-1]  # Mean of gamma
    σγ = params[end]  # Standard deviation of gamma
    
    # Compute probabilities using quadrature
    probs = logit_probabilities_with_quadrature(X, Z, β, μγ, σγ)
    
    # Compute the likelihood
    likelihood = sum(log.(probs[CartesianIndex(i, y[i])] for i in 1:length(y)))
    return -likelihood  # Negative for minimization
end

# Initial guess for parameters
β_init = randn(size(X, 2))
μγ_init = 0.0
σγ_init = 1.0
params_init = vcat(β_init, μγ_init, σγ_init)

# Use Optim with automatic differentiation to minimize the log-likelihood
result = optimize(log_likelihood_quadrature, params_init, BFGS(), autodiff = :forward)

# Extract the optimized parameter estimates
params_opt = Optim.minimizer(result)

# Print results
println("Optimized parameters: ", params_opt)

# Question 5
# Import necessary packages
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, ForwardDiff, Distributions

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Set up X (individual-specific covariates) and Z (alternative-specific covariates)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

# Monte Carlo parameters
D = 10000  # Number of random draws for Monte Carlo integration
rng = MersenneTwister(1234)  # Fix the random seed for reproducibility

# Define the logit probability function with random gamma using Monte Carlo integration
function logit_probabilities_with_monte_carlo(X, Z, β, μγ, σγ)
    N, J = size(Z)
    utilities = X * β  # Alternative-specific utilities
    
    # Ensure σγ is positive and handle Dual types
    μγ_val = ForwardDiff.value(μγ)  # Extract value from μγ if it's a Dual type
    σγ_val = abs(ForwardDiff.value(σγ))  # Ensure σγ is positive and handle Dual types
    
    # Initialize probability matrix
    probabilities = zeros(N, J)
    
    # Draw D random samples for gamma from N(μγ, σγ)
    gamma_draws = rand(Normal(μγ, σγ), D)
    
    # Loop over random draws
    for d in 1:D
        γ_d = gamma_draws[d]
        exp_utilities = exp.(utilities .+ γ_d * (Z .- Z[:, J]))  # Update utilities with random gamma
        denom = 1 .+ sum(exp_utilities[:, 1:J-1], dims=2)
        temp_probs = hcat(exp_utilities[:, 1:J-1] ./ denom, 1 ./ denom)
        
        # Accumulate the probabilities
        probabilities += temp_probs
    end
    
    # Average over the random draws
    probabilities /= D
    return probabilities
end

# Likelihood function (negative log-likelihood) with Monte Carlo integration
function log_likelihood_monte_carlo(params)
    β = params[1:size(X, 2)]  # Beta coefficients
    μγ = params[end-1]  # Mean of gamma
    σγ = params[end]  # Standard deviation of gamma
    
    # Compute probabilities using Monte Carlo integration
    probs = logit_probabilities_with_monte_carlo(X, Z, β, μγ, σγ)
    
    # Compute the likelihood
    likelihood = sum(log.(probs[CartesianIndex(i, y[i])] for i in 1:length(y)))
    return -likelihood  # Negative for minimization
end

# Initial guess for parameters
β_init = randn(size(X, 2))
μγ_init = 0.0
σγ_init = 1.0
params_init = vcat(β_init, μγ_init, σγ_init)

# Use Optim with automatic differentiation to minimize the log-likelihood
result = optimize(log_likelihood_monte_carlo, params_init, BFGS(), autodiff = :forward)

# Extract the optimized parameter estimates
params_opt = Optim.minimizer(result)

# Print results
println("Optimized parameters: ", params_opt)


# Question 6
# Import necessary packages
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, ForwardDiff, Distributions

# Load the dataset
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return df
end

# Monte Carlo integration method (updated to handle Dual types)
function logit_probabilities_with_monte_carlo(X, Z, β, μγ, σγ, D)
    N, J = size(Z)
    utilities = X * β

    # Ensure σγ is positive and handle Dual types
    μγ_val = ForwardDiff.value(μγ)  # Extract value from μγ if it's a Dual type
    σγ_val = abs(ForwardDiff.value(σγ))  # Ensure σγ is positive and handle Dual types

    # Initialize probability matrix
    probabilities = zeros(N, J)
    
    # Draw D random samples for gamma from N(μγ, σγ)
    gamma_draws = rand(Normal(μγ_val, σγ_val), D)
    
    # Loop over random draws
    for d in 1:D
        γ_d = gamma_draws[d]
        exp_utilities = exp.(utilities .+ γ_d * (Z .- Z[:, J]))
        denom = 1 .+ sum(exp_utilities[:, 1:J-1], dims=2)
        temp_probs = hcat(exp_utilities[:, 1:J-1] ./ denom, 1 ./ denom)
        probabilities += temp_probs
    end
    
    probabilities /= D
    return probabilities
end

# Likelihood function using Monte Carlo integration (updated to handle Dual types)
function log_likelihood_monte_carlo(params, X, Z, y, D)
    β = params[1:size(X, 2)]  # Beta coefficients
    μγ = params[end-1]  # Mean of gamma
    σγ = params[end]  # Standard deviation of gamma
    
    # Compute probabilities using Monte Carlo integration
    probs = logit_probabilities_with_monte_carlo(X, Z, β, μγ, σγ, D)
    
    likelihood = sum(log.(probs[CartesianIndex(i, y[i])] for i in 1:length(y)))
    
    println("Log Likelihood: ", likelihood)
    
    return -likelihood  # Negative for minimization
end

# Main function to run the estimation and print results
function estimate_logit_model(method::Symbol, K::Int=7, D::Int=10000)
    # Load data
    df = load_data()
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    # Initial parameter guesses
    β_init = randn(size(X, 2))
    μγ_init = 0.0
    σγ_init = 1.0
    params_init = vcat(β_init, μγ_init, σγ_init)

    # Optimize based on the method selected
    if method == :monte_carlo
        println("Estimating model using Monte Carlo Integration with $D draws...")
        result = optimize(p -> log_likelihood_monte_carlo(p, X, Z, y, D), params_init, BFGS(), autodiff = :forward)
    else
        error("Unknown method: choose :monte_carlo")
    end

    # Extract and print results
    params_opt = Optim.minimizer(result)
    println("Optimized parameters: ", params_opt)
    
    return params_opt
end

# Example call
# estimate_logit_model(:monte_carlo)  # Estimate using Monte Carlo with default 10,000 draws

# Question 7
# Import required testing package
using Test

# Import the main code
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, ForwardDiff, Distributions

# Define a simple test function
function run_tests()

    # Test 1: Test if data loading works correctly
    @testset "Test load_data function" begin
        df = load_data()
        @test !isempty(df)  # Check if the dataframe is not empty
        @test size(df, 1) > 0  # Check if there are rows in the dataframe
        @test "age" in names(df)  # Check if 'age' column exists
    end

    # Test 2: Test Monte Carlo probabilities function
    @testset "Test logit_probabilities_with_monte_carlo function" begin
        X = [1.0 0.5 0.3; 1.0 0.2 0.4]
        Z = [2.0 1.0; 1.5 0.5]
        β = [0.2, 0.1, 0.05]
        μγ = 0.0
        σγ = 1.0
        D = 100
        probabilities = logit_probabilities_with_monte_carlo(X, Z, β, μγ, σγ, D)
        @test size(probabilities) == (2, 2)  # Check if probabilities matrix has correct dimensions
        @test all(probabilities .>= 0.0)  # Check if all probabilities are non-negative
        @test all(probabilities .<= 1.0)  # Check if all probabilities are less than or equal to 1
    end

    # Test 3: Test log likelihood function
    @testset "Test log_likelihood_monte_carlo function" begin
        X = [1.0 0.5 0.3; 1.0 0.2 0.4]
        Z = [2.0 1.0; 1.5 0.5]
        y = [1, 2]
        β = [0.2, 0.1, 0.05]
        μγ = 0.0
        σγ = 1.0
        params = vcat(β, μγ, σγ)
        D = 100
        ll = log_likelihood_monte_carlo(params, X, Z, y, D)
        
        @test ll < 1000  # Check that likelihood is not excessively positive
    end

    # Test 4: Test the overall model estimation function
    @testset "Test estimate_logit_model function" begin
        params_opt = estimate_logit_model(:monte_carlo, 7, 100)
        @test length(params_opt) == 5  # Check if there are 5 optimized parameters (3 β + μγ + σγ)
        @test params_opt[end] > 0  # Check if σγ is positive
    end

end

# Run the tests
run_tests()

