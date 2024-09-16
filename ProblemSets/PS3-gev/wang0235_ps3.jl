# Question 1
using CSV, DataFrames, GLM, LinearAlgebra, Optim, Random, Statistics, HTTP

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Convert X to Float64 to avoid type mismatch
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

function multinomial_logit(X, Z, β, γ)
    J = 8 # Number of occupation choices
    N = size(X, 1) # Number of individuals
    
    P = zeros(N, J)
    
    for i in 1:N
        for j in 1:(J-1)
            Z_diff = Z[i, j] - Z[i, J] # Differencing Z's as per the instruction
            numerator = exp(dot(X[i, :], β[j, :]) + γ * Z_diff)
            denominator = 1 + sum([exp(dot(X[i, :], β[k, :]) + γ * (Z[i, k] - Z[i, J])) for k in 1:(J-1)])
            P[i, j] = numerator / denominator
        end
        P[i, J] = 1 / (1 + sum([exp(dot(X[i, :], β[k, :]) + γ * (Z[i, k] - Z[i, J])) for k in 1:(J-1)]))
    end
    
    return P
end


# Initial parameter guesses (random initialization)
β_initial = randn(size(X, 2), J-1)
γ_initial = randn(1)
initial_params = vcat(vec(β_initial), γ_initial)

# Minimize the negative log-likelihood
result = optimize(params -> log_likelihood(params, X, Z, y, J), initial_params, BFGS())

# Extract the estimated parameters
estimated_params = result.minimizer
β_hat = reshape(estimated_params[1:end-1], size(X, 2), J-1)
γ_hat = estimated_params[end]

println("Estimated β coefficients: ", β_hat)
println("Estimated γ coefficient: ", γ_hat)


# Question 2
# The estimated value of 𝛾 (-0.094) being negative implies that as the value of the alternative-specific variable in 𝑍 (such as wage difference) increases, the probability of choosing that occupation decreases. 
# In other words, higher wage differences reduce the likelihood of someone choosing that occupation. This negative relationship indicates that wage differences have a deterrent effect on occupational choice.


# Question 3
using CSV, DataFrames, Optim, LinearAlgebra, HTTP

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Define the variables
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Define nests
WC = [1, 2, 3]  # White collar occupations
BC = [4, 5, 6, 7]  # Blue collar occupations
Other = [8]  # Other

# Log-likelihood function for nested logit
function log_likelihood_nested(params, X, Z, y, WC, BC, Other)
    # Extract parameters
    β_WC = params[1:3]
    β_BC = params[4:6]
    γ = params[7]
    λ_WC = params[8]
    λ_BC = params[9]
    
    ll = 0.0
    
    for i in 1:size(X, 1)
        # Calculate sums for the white collar and blue collar nests
        sum_WC = sum(exp((dot(X[i, :], β_WC) + γ * (Z[i, j] - Z[i, 8])) / λ_WC) for j in WC)
        sum_BC = sum(exp((dot(X[i, :], β_BC) + γ * (Z[i, j] - Z[i, 8])) / λ_BC) for j in BC)
        
        # Ensure sum_WC and sum_BC are scalars, and then raise to the power of λ
        denom = sum_WC^λ_WC + sum_BC^λ_BC
        
        # Add likelihood contributions
        if y[i] in WC
            ll += log(exp((dot(X[i, :], β_WC) + γ * (Z[i, y[i]] - Z[i, 8])) / λ_WC) / denom)
        elseif y[i] in BC
            ll += log(exp((dot(X[i, :], β_BC) + γ * (Z[i, y[i]] - Z[i, 8])) / λ_BC) / denom)
        elseif y[i] in Other
            ll += log(1 / denom)
        end
    end
    
    return -ll  # Return negative log-likelihood
end

# Initial parameter guess (flattened vector)
initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

# Optimize the negative log-likelihood
result = optimize(params -> log_likelihood_nested(params, X, Z, y, WC, BC, Other), initial_params, BFGS())

# Extract the estimated parameters
estimated_params = result.minimizer
β_WC_hat = estimated_params[1:3]
β_BC_hat = estimated_params[4:6]
γ_hat = estimated_params[7]
λ_WC_hat = estimated_params[8]
λ_BC_hat = estimated_params[9]

println("Estimated β_WC: ", β_WC_hat)
println("Estimated β_BC: ", β_BC_hat)
println("Estimated γ: ", γ_hat)
println("Estimated λ_WC: ", λ_WC_hat)
println("Estimated λ_BC: ", λ_BC_hat)


# Question 4
# Wrap function for Q1
using CSV, DataFrames, GLM, LinearAlgebra, Optim, Random, Statistics, HTTP
function estimate_multinomial_logit()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Convert X to Float64 to avoid type mismatch
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    function log_likelihood(params, X, Z, y, J)
        β = reshape(params[1:end-1], size(X, 2), J-1)
        γ = params[end]
        ll = 0.0

        for i in 1:size(X, 1)
            denom = 1 + sum(exp(dot(X[i, :], β[:, j]) + γ * (Z[i, j] - Z[i, J])) for j in 1:(J-1))
            for j in 1:(J-1)
                if y[i] == j
                    prob = exp(dot(X[i, :], β[:, j]) + γ * (Z[i, j] - Z[i, J])) / denom
                    ll += log(prob)
                end
            end
            if y[i] == J
                prob = 1 / denom
                ll += log(prob)
            end
        end

        return -ll  # Return negative log-likelihood for minimization
    end

    J = 8
    # Initial parameter guesses (random initialization)
    β_initial = randn(size(X, 2), J-1)
    γ_initial = randn(1)
    initial_params = vcat(vec(β_initial), γ_initial)

    # Minimize the negative log-likelihood
    result = optimize(params -> log_likelihood(params, X, Z, y, J), initial_params, BFGS())

    # Extract the estimated parameters
    estimated_params = result.minimizer
    β_hat = reshape(estimated_params[1:end-1], size(X, 2), J-1)
    γ_hat = estimated_params[end]

    println("Estimated β coefficients: ", β_hat)
    println("Estimated γ coefficient: ", γ_hat)
end

# Wrape function for Q3
using CSV, DataFrames, Optim, LinearAlgebra, HTTP
function estimate_nested_logit()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Define the variables
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # Define nests
    WC = [1, 2, 3]  # White collar occupations
    BC = [4, 5, 6, 7]  # Blue collar occupations
    Other = [8]  # Other

    # Log-likelihood function for nested logit
    function log_likelihood_nested(params, X, Z, y, WC, BC, Other)
        β_WC = params[1:3]
        β_BC = params[4:6]
        γ = params[7]
        λ_WC = params[8]
        λ_BC = params[9]

        ll = 0.0

        for i in 1:size(X, 1)
            # Calculate sums for the white collar and blue collar nests
            sum_WC = sum(exp((dot(X[i, :], β_WC) + γ * (Z[i, j] - Z[i, 8])) / λ_WC) for j in WC)
            sum_BC = sum(exp((dot(X[i, :], β_BC) + γ * (Z[i, j] - Z[i, 8])) / λ_BC) for j in BC)

            # Ensure sum_WC and sum_BC are scalars, and then raise to the power of λ
            denom = sum_WC^λ_WC + sum_BC^λ_BC

            if y[i] in WC
                ll += log(exp((dot(X[i, :], β_WC) + γ * (Z[i, y[i]] - Z[i, 8])) / λ_WC) / denom)
            elseif y[i] in BC
                ll += log(exp((dot(X[i, :], β_BC) + γ * (Z[i, y[i]] - Z[i, 8])) / λ_BC) / denom)
            elseif y[i] in Other
                ll += log(1 / denom)
            end
        end

        return -ll  # Return negative log-likelihood
    end

    # Initial parameter guess (flattened vector)
    initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    # Optimize the negative log-likelihood
    result = optimize(params -> log_likelihood_nested(params, X, Z, y, WC, BC, Other), initial_params, BFGS())

    # Extract the estimated parameters
    estimated_params = result.minimizer
    β_WC_hat = estimated_params[1:3]
    β_BC_hat = estimated_params[4:6]
    γ_hat = estimated_params[7]
    λ_WC_hat = estimated_params[8]
    λ_BC_hat = estimated_params[9]

    println("Estimated β_WC: ", β_WC_hat)
    println("Estimated β_BC: ", β_BC_hat)
    println("Estimated γ: ", γ_hat)
    println("Estimated λ_WC: ", λ_WC_hat)
    println("Estimated λ_BC: ", λ_BC_hat)
end

# Call the multinomial logit estimation
println("Multinomial Logit Estimation:")
estimate_multinomial_logit()

# Call the nested logit estimation
println("\nNested Logit Estimation:")
estimate_nested_logit()

# Question 5
using Test
using Test

@testset "Multinomial Logit Model Tests" begin
    println("Running tests for Multinomial Logit model...")

    try
        estimate_multinomial_logit()

        J = 8
        β_hat = randn(3, J-1)
        γ_hat = randn()

        @test size(β_hat) == (3, J-1) 
        @test γ_hat isa Float64  

        println("Multinomial Logit model tests passed!")
    catch e
        println("Error during Multinomial Logit tests: ", e)
        @test false 
    end
end

@testset "Nested Logit Model Tests" begin
    println("Running tests for Nested Logit model...")

    try
        estimate_nested_logit()

        β_WC_hat = randn(3)
        β_BC_hat = randn(3)
        γ_hat = randn()
        λ_WC_hat = randn()
        λ_BC_hat = randn()

        @test length(β_WC_hat) == 3 
        @test length(β_BC_hat) == 3 
        @test γ_hat isa Float64  
        @test λ_WC_hat isa Float64 
        @test λ_BC_hat isa Float64  

        println("Nested Logit model tests passed!")
    catch e
        println("Error during Nested Logit tests: ", e)
        @test false  
    end
end

