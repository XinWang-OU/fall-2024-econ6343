# Question 1

using Markdown

md_text = Markdown.parse("""
## Moment Conditions
The moment conditions are defined as:

\\[
g(\\beta) = \\frac{1}{N} X^T (y - X\\beta)
\\]

## GMM Objective Function
The objective function is:

\\[
J(\\beta) = g(\\beta)^T W g(\\beta)
\\]

where \\(W = I\\), the identity matrix, which makes GMM equivalent to OLS in this context.
""")

println(md_text)

using Optim, LinearAlgebra, DataFrames, CSV, HTTP

# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Define X and y
# X contains a constant term, age, race (whether white), and collgrad (whether graduated from college)
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1

# Step 3: Define the GMM objective function
function gmm_objective(beta, X, y)
    residuals = y - X * beta  # Residuals
    g = X' * residuals / size(X, 1)  # Compute the moment function
    return g' * g  # Return the objective function value J(beta)
end

# Step 4: Estimate using Optim
startval = rand(size(X, 2))  # Random starting values
result = optimize(b -> gmm_objective(b, X, y), startval, LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))

# Print GMM estimates
println("GMM Estimates: ", result.minimizer)

# Step 5: Compare with OLS estimates
# Closed-form OLS estimator: β_hat_ols = (X'X)^(-1) X'y
beta_hat_ols = inv(X' * X) * X' * y
println("OLS Estimates: ", beta_hat_ols)


# Question 2
# (a)
using Optim, DataFrames, CSV, HTTP

# Load and clean the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
df = dropmissing(df, :occupation)
df[df.occupation .== 8, :occupation] .= 7  # Recode occupation

# Define X (covariates) and y (response)
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.occupation

# Define the log-likelihood function
function log_likelihood(beta, X, y, J)
    N = size(X, 1)  # Number of observations
    K = size(X, 2)  # Number of covariates
    beta_matrix = reshape(beta, K, J - 1)  # Reshape beta vector into K × (J - 1)
    logL = 0  # Initialize log-likelihood
    for i in 1:N
        Xi = X[i, :]  # Covariates for individual i
        denom = 1 + sum(exp.(Xi' * beta_matrix))  # Element-wise exponentiation
        for j in 1:(J - 1)
            if y[i] == j
                logL += Xi' * beta_matrix[:, j] - log(denom)
            end
        end
    end
    return -logL  # Return negative log-likelihood to minimize
end


# Initial starting values for beta
J = 7  # Number of occupation categories (as per recoding)
K = size(X, 2)  # Number of covariates
beta_start = rand(K * (J - 1))  # Random starting values

# Optimize log-likelihood
result = optimize(b -> log_likelihood(b, X, y, J), beta_start, LBFGS())

# Output the estimated parameters
println("MLE Estimates for Multinomial Logit Model: ", result.minimizer)

# (b)
using LinearAlgebra, Optim

# Define the moment conditions
function moment_conditions(beta, X, y, J)
    N = size(X, 1)
    K = size(X, 2)
    beta_matrix = reshape(beta, K, J - 1)
    
    g = zeros(N, J - 1)  # Moment conditions matrix
    
    for i in 1:N
        Xi = X[i, :]
        denom = 1 + sum(exp.(Xi' * beta_matrix))
        probs = exp.(Xi' * beta_matrix) / denom  # Predicted probabilities
        
        for j in 1:(J - 1)
            g[i, j] = (y[i] == j ? 1.0 : 0.0) - probs[j]
        end
    end
    
    return vec(g)  # Stack the moment conditions into a vector
end

# Define the GMM objective function
function gmm_objective(beta, X, y, J, W)
    g = moment_conditions(beta, X, y, J)
    return g' * W * g  # GMM objective: g' * W * g
end

# Initialize parameters from MLE
beta_start = [0.7873488037624246, 0.41049671481702565, 0.7589762379069055, 2.377032851437638,
              0.4790808281949961, 0.4081192617593853, 1.3833898146335557, 1.5714737922958444,
              1.4309584782199114, 0.43300171189587744, 0.5291122826281772, 0.459647830102284,
              -1.5766447794448872, 0.43863942770123704, 1.4261496376274212, 1.0218781440605327,
              -0.6298018130017935, 0.4286295362810589, -0.012394955579424329, 0.4064196020024682,
              1.0569803515170648, 0.43584370287130386, -0.577240377789691, -1.944817386709371]

# Identity matrix as weighting matrix W
W = I  # N × N identity matrix

# Run GMM estimation using the Optim package
result_gmm = optimize(b -> gmm_objective(b, X, y, J, W), beta_start, LBFGS())

# Output the GMM estimates
println("GMM Estimates with MLE Starting Values: ", result_gmm.minimizer)

# (c)
using Random, Distributions

# use a random seed for reproducibility
# assume the seed is selacted from -1 to 1
random_start = rand(Uniform(-1, 1), length(beta_start))

# run GMM estimation
result_gmm_random = optimize(b -> gmm_objective(b, X, y, J, W), random_start, LBFGS())

println("GMM Estimates with Random Starting Values: ", result_gmm_random.minimizer)

println("Comparison of GMM estimates with MLE starting values and random starting values:")
println("From MLE starting values: ", result_gmm.minimizer)
println("From random starting values: ", result_gmm_random.minimizer)

# The significant differences between the estimates from part (b) and part (c) strongly suggest that the GMM objective function is not globally concave. 
# The choice of starting values has a substantial impact on the final estimates.
# This indicates that the optimization surface likely contains multiple local minima or regions where the optimizer struggles to find the global minimum.


# Question 3
using Random, LinearAlgebra

# 3(a): Generate X and beta
function generate_data(N, J, K)
    # N: Sample size (number of observations)
    # J: Dimension of the choice set (number of choices)
    # K: Number of covariates in X

    # Step 1: Generate random X matrix with N rows and K columns
    X = randn(N, K)  # Random normal values for covariates (N × K)

    # Step 2: Set values for beta ensuring conformability with X and J
    beta = randn(K, J - 1)  # Random beta for each choice except the base category (K × (J - 1))

    return X, beta
end

# 3(b): Generate the N × J matrix of choice probabilities P
function calculate_choice_probabilities(X, beta, J)
    N, K = size(X)
    P = zeros(N, J)  # Initialize probability matrix (N × J)

    # Compute the utility for each choice j = 1, 2, ..., J-1 (without base category)
    for i in 1:N
        util = X[i, :]' * beta  # Use transpose to ensure correct dimensions (1 × (J-1))
        
        denom = 1 + sum(exp.(util))  # Denominator including base category
        P[i, 1:J-1] .= exp.(util)' ./ denom  # Explicitly transpose util to ensure 1-dimensional broadcast
        P[i, J] = 1 / denom  # Probability for the base category (choice J)
    end

    return P
end

# 3(c): Generate Y based on the choice probabilities
function generate_choices(P)
    N, J = size(P)
    Y = zeros(Int, N)  # Initialize Y as an N × 1 vector of zeros (for storing choices)

    # Step through each observation and generate a choice based on probabilities
    for i in 1:N
        # Draw a random number between 0 and 1
        epsilon = rand()

        # Compute cumulative probabilities and find the choice
        cumulative_prob = 0.0
        for j in 1:J
            cumulative_prob += P[i, j]
            if epsilon < cumulative_prob
                Y[i] = j  # Assign choice j to Y[i] if cumulative probability exceeds epsilon
                break
            end
        end
    end

    return Y
end

# Define parameters
N = 100  # Sample size
J = 4    # Number of choices
K = 3    # Number of covariates

# Generate data
X, beta = generate_data(N, J, K)

# Calculate choice probabilities
P = calculate_choice_probabilities(X, beta, J)

# Generate choices based on probabilities
Y = generate_choices(P)

# Output the first 10 choices
println("Generated choices (first 10 observations):")
println(Y[1:10])

# 3(d): Generate the preference shocks ε
function generate_preference_shocks(N)
    # Draw the preference shocks ε as an N × 1 vector of U[0,1] random numbers
    ε = rand(N)  # Random uniform values between 0 and 1 (N × 1)

    return ε
end

# Example usage for part 3(d)
ε = generate_preference_shocks(N)
println("Preference shocks (first 10 observations):")
println(ε[1:10])

# 3(e): Generate Y based on the choice probabilities and preference shocks
function generate_choices_with_shocks(P, ε)
    N, J = size(P)
    Y = zeros(Int, N)  # Initialize Y as an N × 1 vector of zeros (for storing choices)

    # Step through each observation and generate a choice based on probabilities and shocks
    for i in 1:N
        # Compute cumulative probabilities
        cumulative_prob = 0.0
        for j in 1:J
            cumulative_prob += P[i, j]
            if ε[i] < cumulative_prob
                Y[i] = j  # Assign choice j to Y[i] if cumulative probability exceeds ε[i]
                break
            end
        end
    end

    return Y
end

# Example usage for part 3(e)
Y_new = generate_choices_with_shocks(P, ε)
println("Generated choices with preference shocks (first 10 observations):")
println(Y_new[1:10])

using Distributions

# 3(f): Generate Y using T1EV-distributed shocks
function generate_choices_T1EV(X, beta, J)
    N, K = size(X)
    Y = zeros(Int, N)  # Initialize Y as an N × 1 vector of zeros (for storing choices)

    # Create a Type-1 Extreme Value distribution object
    T1EV_dist = Gumbel(0, 1)  # Gumbel distribution is equivalent to T1EV

    # Step through each observation and generate a choice
    for i in 1:N
        utilities = zeros(J)  # Initialize utility for each choice
        
        # Compute utility for each choice j
        for j in 1:(J-1)
            utilities[j] = dot(X[i, :], beta[:, j]) + rand(T1EV_dist)  # Xβ + T1EV shock
        end
        
        # Utility for the base category (J-th choice)
        utilities[J] = rand(T1EV_dist)  # Only T1EV shock for the base category
        
        # Assign choice as the one with the highest utility
        Y[i] = argmax(utilities)
    end

    return Y
end

# Example usage for part 3(f)
Y_T1EV = generate_choices_T1EV(X, beta, J)
println("Generated choices with T1EV-distributed shocks (first 10 observations):")
println(Y_T1EV[1:10])





# Question 5
using Random, Optim, LinearAlgebra

# Step 1: Define the moment calculation function for observed data
function calculate_data_moments(Y)
    N = length(Y)
    moments = zeros(N)  # Store moments as a vector
    for i in 1:N
        moments[i] = Y[i]  # For simplicity, just store the choices
    end
    return moments
end

# Step 2: Simulate the data moments based on guessed parameters (beta)
function simulate_moments(X, beta, J, D)
    N, K = size(X)
    simulated_moments = zeros(N, D)  # Store moments from D simulations
    beta = reshape(beta, K, J-1)  # Reshape beta back to matrix form
    for d in 1:D
        P = calculate_choice_probabilities(X, beta, J)
        Y_sim = generate_choices(P)
        simulated_moments[:, d] .= Y_sim
    end
    return mean(simulated_moments, dims=2)  # Return average across simulations
end

# Step 3: Objective function for SMM
function smm_objective(flat_beta, X, Y, J, D)
    K = size(X, 2)
    # Reshape the flat beta vector into K × (J-1) matrix
    beta = reshape(flat_beta, K, J-1)
    
    # 1. Calculate data moments
    data_moments = calculate_data_moments(Y)
    
    # 2. Simulate moments based on current guess of beta
    simulated_moments = simulate_moments(X, beta, J, D)
    
    # 3. Compute the difference between data moments and simulated moments
    moment_diff = data_moments .- simulated_moments
    
    # 4. Define the weighting matrix (identity for simplicity)
    W = I  # Identity matrix
    
    # 5. Compute the objective function (quadratic form)
    J_val = dot(moment_diff, W * moment_diff)  # Ensure scalar result
    
    return J_val
end

# Step 4: Run the optimization process to minimize the SMM objective function
function smm_estimate(X, Y, J, D)
    K = size(X, 2)
    initial_beta = randn(K * (J - 1))  # Initial guess for beta, flattened as a vector
    result = optimize(flat_beta -> smm_objective(flat_beta, X, Y, J, D), initial_beta, LBFGS())
    
    # Reshape the minimizer back to K × (J-1) form
    estimated_beta = reshape(result.minimizer, K, J-1)
    
    return estimated_beta
end

# Parameters
N = 100  # Sample size
J = 4    # Number of choices
K = 3    # Number of covariates
D = 1000  # Number of simulations

# Generate data
X, beta_true = generate_data(N, J, K)
P = calculate_choice_probabilities(X, beta_true, J)
Y = generate_choices(P)

# Run SMM estimation
estimated_beta = smm_estimate(X, Y, J, D)
println("Estimated beta:")
println(estimated_beta)



# Question 6,7
using Markdown, Optim, LinearAlgebra, DataFrames, CSV, HTTP, Random, Distributions

function main()
    # Question 1

    md_text = Markdown.parse("""
    ## Moment Conditions
    The moment conditions are defined as:

    \\[
    g(\\beta) = \\frac{1}{N} X^T (y - X\\beta)
    \\]

    ## GMM Objective Function
    The objective function is:

    \\[
    J(\\beta) = g(\\beta)^T W g(\\beta)
    \\]

    where \\(W = I\\), the identity matrix, which makes GMM equivalent to OLS in this context.
    """)

    println(md_text)

    # Step 1: Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Step 2: Define X and y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    # Step 3: Define the GMM objective function
    function gmm_objective(beta, X, y)
        residuals = y - X * beta
        g = X' * residuals / size(X, 1)
        return g' * g
    end

    # Step 4: Estimate using Optim
    startval = rand(size(X, 2))
    result = optimize(b -> gmm_objective(b, X, y), startval, LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))

    println("GMM Estimates: ", result.minimizer)

    # Step 5: Compare with OLS estimates
    beta_hat_ols = inv(X' * X) * X' * y
    println("OLS Estimates: ", beta_hat_ols)

    # Question 2(a)
    df = dropmissing(df, :occupation)
    df[df.occupation .== 8, :occupation] .= 7

    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    function log_likelihood(beta, X, y, J)
        N = size(X, 1)
        K = size(X, 2)
        beta_matrix = reshape(beta, K, J - 1)
        logL = 0
        for i in 1:N
            Xi = X[i, :]
            denom = 1 + sum(exp.(Xi' * beta_matrix))
            for j in 1:(J - 1)
                if y[i] == j
                    logL += Xi' * beta_matrix[:, j] - log(denom)
                end
            end
        end
        return -logL
    end

    J = 7
    K = size(X, 2)
    beta_start = rand(K * (J - 1))

    result = optimize(b -> log_likelihood(b, X, y, J), beta_start, LBFGS())

    println("MLE Estimates for Multinomial Logit Model: ", result.minimizer)

    # (b) Moment Conditions and GMM

    function moment_conditions(beta, X, y, J)
        N = size(X, 1)
        K = size(X, 2)
        beta_matrix = reshape(beta, K, J - 1)

        g = zeros(N, J - 1)

        for i in 1:N
            Xi = X[i, :]
            denom = 1 + sum(exp.(Xi' * beta_matrix))
            probs = exp.(Xi' * beta_matrix) / denom

            for j in 1:(J - 1)
                g[i, j] = (y[i] == j ? 1.0 : 0.0) - probs[j]
            end
        end

        return vec(g)
    end

    function gmm_objective(beta, X, y, J, W)
        g = moment_conditions(beta, X, y, J)
        return g' * W * g
    end

    # Initialize parameters from MLE
    beta_start = result.minimizer

    # Identity matrix as weighting matrix W
    W = I

    # Run GMM estimation using the Optim package
    result_gmm = optimize(b -> gmm_objective(b, X, y, J, W), beta_start, LBFGS())

    println("GMM Estimates with MLE Starting Values: ", result_gmm.minimizer)

    # (c) GMM with Random Starting Values
    random_start = rand(Uniform(-1, 1), length(beta_start))

    result_gmm_random = optimize(b -> gmm_objective(b, X, y, J, W), random_start, LBFGS())

    println("GMM Estimates with Random Starting Values: ", result_gmm_random.minimizer)

    # Question 3
    # 3(a): Generate X and beta
    function generate_data(N, J, K)
        X = randn(N, K)
        beta = randn(K, J - 1)
        return X, beta
    end

    # 3(b): Generate the N × J matrix of choice probabilities P
    function calculate_choice_probabilities(X, beta, J)
        N, K = size(X)
        P = zeros(N, J)

        for i in 1:N
            util = X[i, :]' * beta
            denom = 1 + sum(exp.(util))
            P[i, 1:J-1] .= exp.(util)' ./ denom
            P[i, J] = 1 / denom
        end

        return P
    end

    # 3(c): Generate Y based on the choice probabilities
    function generate_choices(P)
        N, J = size(P)
        Y = zeros(Int, N)

        for i in 1:N
            epsilon = rand()
            cumulative_prob = 0.0
            for j in 1:J
                cumulative_prob += P[i, j]
                if epsilon < cumulative_prob
                    Y[i] = j
                    break
                end
            end
        end

        return Y
    end

    # Parameters
    N = 100
    J = 4
    K = 3

    # Generate data
    X, beta = generate_data(N, J, K)

    # Calculate choice probabilities
    P = calculate_choice_probabilities(X, beta, J)

    # Generate choices based on probabilities
    Y = generate_choices(P)

    println("Generated choices (first 10 observations):")
    println(Y[1:10])

    # 3(d): Generate the preference shocks ε
    function generate_preference_shocks(N)
        ε = rand(N)
        return ε
    end

    # Example usage for part 3(d)
    ε = generate_preference_shocks(N)
    println("Preference shocks (first 10 observations):")
    println(ε[1:10])

    # 3(e): Generate Y based on the choice probabilities and preference shocks
    function generate_choices_with_shocks(P, ε)
        N, J = size(P)
        Y = zeros(Int, N)

        for i in 1:N
            cumulative_prob = 0.0
            for j in 1:J
                cumulative_prob += P[i, j]
                if ε[i] < cumulative_prob
                    Y[i] = j
                    break
                end
            end
        end

        return Y
    end

    # Example usage for part 3(e)
    Y_new = generate_choices_with_shocks(P, ε)
    println("Generated choices with preference shocks (first 10 observations):")
    println(Y_new[1:10])

    # 3(f): Generate Y using T1EV-distributed shocks
    function generate_choices_T1EV(X, beta, J)
        N, K = size(X)
        Y = zeros(Int, N)

        T1EV_dist = Gumbel(0, 1)

        for i in 1:N
            utilities = zeros(J)
            for j in 1:(J-1)
                utilities[j] = dot(X[i, :], beta[:, j]) + rand(T1EV_dist)
            end
            utilities[J] = rand(T1EV_dist)
            Y[i] = argmax(utilities)
        end

        return Y
    end

    # Example usage for part 3(f)
    Y_T1EV = generate_choices_T1EV(X, beta, J)
    println("Generated choices with T1EV-distributed shocks (first 10 observations):")
    println(Y_T1EV[1:10])

    # Question 5
    function calculate_data_moments(Y)
        N = length(Y)
        moments = zeros(N)
        for i in 1:N
            moments[i] = Y[i]
        end
        return moments
    end

    function simulate_moments(X, beta, J, D)
        N, K = size(X)
        simulated_moments = zeros(N, D)
        beta = reshape(beta, K, J-1)
        for d in 1:D
            P = calculate_choice_probabilities(X, beta, J)
            Y_sim = generate_choices(P)
            simulated_moments[:, d] .= Y_sim
        end
        return mean(simulated_moments, dims=2)
    end

    function smm_objective(flat_beta, X, Y, J, D)
        K = size(X, 2)
        beta = reshape(flat_beta, K, J-1)
        data_moments = calculate_data_moments(Y)
        simulated_moments = simulate_moments(X, beta, J, D)
        moment_diff = data_moments .- simulated_moments
        W = I
        J_val = dot(moment_diff, W * moment_diff)
        return J_val
    end

    function smm_estimate(X, Y, J, D)
        K = size(X, 2)
        initial_beta = randn(K * (J - 1))
        result = optimize(flat_beta -> smm_objective(flat_beta, X, Y, J, D), initial_beta, LBFGS())
        estimated_beta = reshape(result.minimizer, K, J-1)
        return estimated_beta
    end

    D = 1000
    estimated_beta = smm_estimate(X, Y, J, D)
    println("Estimated beta:")
    println(estimated_beta)
end

# Call the main function
main()

using Test
using LinearAlgebra
using Random
using Distributions

function simulate_y_for_test(X, beta, J)
    N = size(X, 1)
    P = zeros(N, J)
    for i in 1:N
        Xi = X[i, :]
        util = Xi' * beta
        denom = 1 + sum(exp.(util))
        P[i, 1:J-1] .= exp.(util)' ./ denom
        P[i, J] = 1 / denom
    end

    y = zeros(Int, N)
    for i in 1:N
        cumulative_prob = cumsum(P[i, :])
        r = rand()
        y[i] = findfirst(cumulative_prob .>= r)
    end
    return y
end


@testset "Test gmm_objective function" begin

    N = 100
    K = 5
    X = randn(N, K)
    beta_true = randn(K)
    y = X * beta_true + randn(N)

    beta_test = randn(K)
    obj_value = gmm_objective(beta_test, X, y)
    @test isfinite(obj_value)
    @test typeof(obj_value) == Float64
end

@testset "Test log_likelihood function" begin
    N = 100
    K = 5
    J = 4
    X = randn(N, K)
    beta_true = randn(K, J - 1)

    y = simulate_y_for_test(X, beta_true, J)
    
    beta_test = randn(K * (J -1))
    ll_value = log_likelihood(beta_test, X, y, J)
    @test isfinite(ll_value)
    @test typeof(ll_value) == Float64
end

@testset "Test moment_conditions function" begin
    N = 100
    K = 5
    J = 4
    X = randn(N, K)
    beta_true = randn(K, J - 1)
    y = simulate_y_for_test(X, beta_true, J)
    
    beta_test = randn(K * (J -1))
    g = moment_conditions(beta_test, X, y, J)
    @test size(g) == (N * (J -1),)
    @test typeof(g) <: AbstractArray{Float64}
end

@testset "Test gmm_objective for multinomial logit" begin
    N = 100
    K = 5
    J = 4
    X = randn(N, K)
    beta_true = randn(K, J - 1)
    y = simulate_y_for_test(X, beta_true, J)
    beta_test = randn(K * (J -1))
    W = I
    obj_value = gmm_objective(beta_test, X, y, J, W)
    @test isfinite(obj_value)
    @test typeof(obj_value) == Float64
end

@testset "Test generate_data function" begin
    N = 100
    J = 4
    K = 3
    X_test, beta_test = generate_data(N, J, K)
    @test size(X_test) == (N, K)
    @test size(beta_test) == (K, J -1)
    @test typeof(X_test) <: AbstractArray{Float64}
    @test typeof(beta_test) <: AbstractArray{Float64}
end


@testset "Test calculate_choice_probabilities function" begin
    N = 100
    J = 4
    K = 3
    X_test = randn(N, K)
    beta_test = randn(K, J -1)
    P = calculate_choice_probabilities(X_test, beta_test, J)
    @test size(P) == (N, J)
    @test all(P .>= 0) && all(P .<= 1)
    @test all(abs.(sum(P, dims=2) .- 1) .< 1e-8)
end

@testset "Test generate_choices function" begin
    N = 100
    J = 4
    P = rand(N, J)
    P .= P ./ sum(P, dims=2)
    y = generate_choices(P)
    @test length(y) == N
    @test all(y .>= 1) && all(y .<= J)
    @test eltype(y) <: Integer
end

@testset "Test generate_preference_shocks function" begin
    N = 100
    ε = generate_preference_shocks(N)
    @test length(ε) == N
    @test typeof(ε) <: AbstractArray{Float64}
end


@testset "Test generate_choices_with_shocks function" begin
    N = 100
    J = 4
    P = rand(N, J)
    P .= P ./ sum(P, dims=2)
    ε = rand(N)
    y = generate_choices_with_shocks(P, ε)
    @test length(y) == N
    @test all(y .>= 1) && all(y .<= J)
    @test eltype(y) <: Integer
end


@testset "Test generate_choices_T1EV function" begin
    N = 100
    J = 4
    K = 3
    X_test = randn(N, K)
    beta_test = randn(K, J -1)
    y = generate_choices_T1EV(X_test, beta_test, J)
    @test length(y) == N
    @test all(y .>= 1) && all(y .<= J)
    @test eltype(y) <: Integer
end


@testset "Test calculate_data_moments function" begin
    N = 100
    Y = rand(1:4, N)
    moments = calculate_data_moments(Y)
    @test length(moments) == N
    @test typeof(moments) <: AbstractArray{Float64}
end

@testset "Test simulate_moments function" begin
    N = 100
    J = 4
    K = 3
    D = 10
    X_test = randn(N, K)
    beta_test = randn(K * (J -1))
    sim_moments = simulate_moments(X_test, beta_test, J, D)
    @test size(sim_moments) == (N, 1)
    @test typeof(sim_moments) <: AbstractArray{Float64}
end

@testset "Test smm_objective function" begin
    N = 100
    J = 4
    K = 3
    D = 10
    X_test = randn(N, K)
    beta_true = randn(K * (J -1))
    Y = rand(1:J, N)
    obj_value = smm_objective(beta_true, X_test, Y, J, D)
    @test isfinite(obj_value)
    @test typeof(obj_value) == Float64
end


@testset "Test smm_estimate function" begin
    N = 100
    J = 4
    K = 3
    D = 10
    X_test = randn(N, K)
    Y = rand(1:J, N)
    estimated_beta = smm_estimate(X_test, Y, J, D)
    @test size(estimated_beta) == (K, J -1)
    @test typeof(estimated_beta) <: AbstractArray{Float64}
end


