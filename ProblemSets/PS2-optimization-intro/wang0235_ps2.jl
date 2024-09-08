#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Optim
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

using GLM
bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3: Logit likelihood estimation
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function logit(alpha, X, d)
    # Predicted probabilities using the logistic function
    p = 1.0 ./ (1.0 .+ exp.(-X * alpha))
    
    # Log-likelihood calculation
    loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    
    # We need to return the negative log-likelihood because Optim minimizes
    return -loglike
end


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4: Logistic regression using GLM for validation
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Load the required packages
using Optim

# Define the covariates matrix X and dependent variable d (binary outcomes)
X = [ones(size(df, 1), 1) df.age df.race.==1 df.collgrad.==1]
d = df.married .== 1

# Use the LBFGS optimizer to minimize the negative log-likelihood
result_logit = optimize(a -> logit(a, X, d), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))

# Print the estimated coefficients (alpha)
println("Estimated coefficients for logit model: ", result_logit.minimizer)

using GLM

# Prepare the data frame for GLM (ensuring that variables are in the right format)
df.white = df.race .== 1  # Add a new column 'white' (race == 1)
df.married_binary = df.married .== 1  # Create a binary married variable

# Fit the logistic regression model using GLM's glm function
logit_glm = glm(@formula(married_binary ~ age + white + collgrad), df, Binomial(), LogitLink())

# Print the coefficients obtained from GLM
println("Logit model coefficients from GLM: ", coef(logit_glm))

# Conclusion: The estimated coefficients from the manual logit function and GLM are similar, indicating that the manual logit function is correctly implemented.


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5: Multinomial Logit Model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Optim
using DataFrames
using FreqTables
using CSV
using HTTP

# Load the data (if not already loaded in previous steps)
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 1: Data cleaning for the 'occupation' variable
# Recode occupation categories and remove missing values
df = dropmissing(df, :occupation)  # Drop rows where occupation is missing
df[df.occupation .== 8, :occupation] .= 7  # Recode occupations 8, 9, 10, 11, 12, 13 to 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Step 2: Define the design matrix (X) and the dependent variable (y)
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.occupation  # The dependent variable (occupation)

# Step 3: Multinomial logit function
function mlogit(alpha, X, y, J)
    N = size(X, 1)  # Number of observations
    K = size(X, 2)  # Number of covariates
    alpha_matrix = reshape(alpha, K, J-1)  # Reshape parameter vector into matrix (K x (J-1))
    
    # Compute utility for each choice alternative
    utilities = X * alpha_matrix
    
    # Add a column of zeros for the baseline category (reference)
    utilities = hcat(utilities, zeros(N))
    
    # Compute the probabilities using the softmax function
    exp_utilities = exp.(utilities)
    probs = exp_utilities ./ sum(exp_utilities, dims=2)
    
    # Compute the log-likelihood
    loglike = 0.0
    for i in 1:N
        chosen_category = y[i]
        loglike += log(probs[i, chosen_category])
    end
    
    # Return the negative log-likelihood (since Optim is a minimizer)
    return -loglike
end

# Step 4: Setting up the multinomial logit optimization
J = 7  # Number of occupation categories after recoding
K = size(X, 2)  # Number of covariates
initial_alpha = rand(K * (J-1))  # Random initial values for the coefficients

# Optimize the negative log-likelihood
result_mlogit = optimize(a -> mlogit(a, X, y, J), initial_alpha, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000))

# Step 5: Output the estimated coefficients
alpha_hat = result_mlogit.minimizer
println("Estimated coefficients for the multinomial logit model: ", alpha_hat)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6: Wrap all of the previous code into functions
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
# Function to load data
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return df
end

# Function for Question 1: Basic Optimization
function optimize_function()
    f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
    minusf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
    startval = rand(1)
    result = optimize(minusf, startval, LBFGS())
    
    println("Question 1: Optimization")
    println("argmin (minimizer) is ", Optim.minimizer(result)[1])
    println("min is ", Optim.minimum(result))
    
    return result  # Return the optimization result
end


# Function for Question 2: OLS Estimation
function ols_estimation(df)
    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    function ols(beta, X, y)
        ssr = (y .- X * beta)' * (y .- X * beta)
        return ssr
    end

    # OLS using Optim
    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println("Question 2: OLS using Optim")
    println("OLS estimates: ", beta_hat_ols.minimizer)
    
    # OLS using GLM
    df.white = df.race .== 1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println("OLS estimates from GLM: ", coef(bols_lm))
    
    # Return both results for testing purposes
    return (beta_hat_ols.minimizer, coef(bols_lm))
end

# Function for Question 3: Logit Likelihood Estimation
function logit_likelihood(df)
    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    d = df.married .== 1

    function logit(alpha, X, d)
        p = 1.0 ./ (1.0 .+ exp.(-X * alpha))
        loglike = sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
        return -loglike  # We return negative log-likelihood
    end

    # Use LBFGS optimizer to minimize the negative log-likelihood
    result_logit = optimize(a -> logit(a, X, d), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))
    
    println("Question 3: Logit using Optim")
    println("Logit estimates: ", result_logit.minimizer)
    
    return result_logit.minimizer  # Return the estimated coefficients
end

# Function for Question 4: Logit using GLM for Validation
function logit_glm_validation(df)
    df.white = df.race .== 1  # Add a binary 'white' variable
    df.married_binary = df.married .== 1  # Create a binary 'married' variable

    # Fit the logistic regression model using GLM's glm function
    logit_glm = glm(@formula(married_binary ~ age + white + collgrad), df, Binomial(), LogitLink())

    println("Question 4: Logit using GLM")
    println("Logit estimates from GLM: ", coef(logit_glm))
    
    return coef(logit_glm)  # Return the coefficients for testing purposes
end

# Function for Question 5: Multinomial Logit Model
function multinomial_logit(df)
    # Data cleaning for the occupation variable
    df = dropmissing(df, :occupation)
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7
    
    # Define X and y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation
    J = 7  # Number of occupation categories
    K = size(X, 2)  # Number of covariates

    # Multinomial logit function
    function mlogit(alpha, X, y, J)
        N = size(X, 1)
        alpha_matrix = reshape(alpha, K, J-1)
        utilities = X * alpha_matrix
        utilities = hcat(utilities, zeros(N))  # Reference category
        exp_utilities = exp.(utilities)
        probs = exp_utilities ./ sum(exp_utilities, dims=2)
        loglike = 0.0
        for i in 1:N
            loglike += log(probs[i, y[i]])
        end
        return -loglike  # Return negative log-likelihood
    end

    # Optimize the multinomial logit
    initial_alpha = rand(K * (J-1))
    result_mlogit = optimize(a -> mlogit(a, X, y, J), initial_alpha, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000))
    
    println("Question 5: Multinomial Logit using Optim")
    println("Multinomial Logit estimates: ", result_mlogit.minimizer)
    
    return result_mlogit.minimizer  # Return the estimated coefficients
end


# Main Function: Call all functions
function main()
    df = load_data()
    
    optimize_function()
    ols_estimation(df)
    logit_likelihood(df)
    logit_glm_validation(df)
    multinomial_logit(df)
end

# Call the main function
main()


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 7: Unit Tests for all functions
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Test

# Test function for Question 1: Optimization
@testset "Optimization Test" begin
    result = optimize_function()
    @test result != nothing  # Ensure function returns a result
end

# Test function for Question 2: OLS Estimation
@testset "OLS Estimation Test" begin
    df = load_data()
    result = ols_estimation(df)
    
    # Ensure the function returns both sets of OLS estimates
    @test result != nothing  # Ensure the result is not nothing
    @test length(result) == 2  # Ensure two results are returned (Optim and GLM)
    
    # Optionally, check if the results are of the expected size
    @test length(result[1]) == 4  # Optim result should have 4 coefficients
    @test length(result[2]) == 4  # GLM result should have 4 coefficients
end

# Test function for Question 3: Logit Likelihood Estimation
@testset "Logit Likelihood Test" begin
    df = load_data()
    result = logit_likelihood(df)
    
    # Ensure the function returns Logit estimates
    @test result != nothing  # Ensure the result is not nothing
    
    # Optionally, check if the result has the expected number of coefficients (should be 4)
    @test length(result) == 4  # We expect 4 coefficients (Intercept, age, race, collgrad)
end

# Test function for Question 4: Logit using GLM for Validation
@testset "Logit GLM Validation Test" begin
    df = load_data()
    result = logit_glm_validation(df)
    
    # Ensure the function returns Logit GLM estimates
    @test result != nothing  # Ensure the result is not nothing
    
    # Optionally, check if the result has the expected number of coefficients (should be 4)
    @test length(result) == 4  # We expect 4 coefficients (Intercept, age, white, collgrad)
end


# Test function for Question 5: Multinomial Logit Model
@testset "Multinomial Logit Test" begin
    df = load_data()
    result = multinomial_logit(df)
    
    # Ensure the function returns Multinomial Logit estimates
    @test result != nothing  # Ensure the result is not nothing
    
    # Optionally, check if the result has the expected number of coefficients
    # Since there are 4 covariates and 6 categories (J-1), we expect 4 * 6 = 24 coefficients
    @test length(result) == 24  # We expect 24 coefficients
end


