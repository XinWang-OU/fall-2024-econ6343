using HTTP
using CSV
using DataFrames
using Optim
using LinearAlgebra
using Random
using Statistics
using Distributions
using MultivariateStats
using StatsModels
using GLM

include("lgwt.jl")


# Question 1
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
response = HTTP.get(url)
df = CSV.read(IOBuffer(response.body), DataFrame)

# Step 3: Define the regression model formula
formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr)

# Step 4: Estimate the linear regression model
model = lm(formula, df)

# Step 5: Display the regression results
println("Regression Results:")
display(coeftable(model))


# Question 2
asvab_vars = select(df, r"asvab")
cor_matrix = cor(Matrix(asvab_vars))
println("Correlation Matrix for ASVAB Variables:")
display(cor_matrix)


# Question 3
extended_formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK)
extended_model = lm(extended_formula, df)
println("Extended Regression Results with ASVAB Variables:")
display(coeftable(extended_model))
# Yes, it would likely be problematic because the ASVAB variables are highly correlated with each other, which can lead to multicollinearity issues.


# Question 4
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
response = HTTP.get(url)
df = CSV.read(IOBuffer(response.body), DataFrame)
# Select ASVAB variables and ensure they are in a matrix format (J x N)
asvab_vars = select(df, r"asvab")
asvabMat = Matrix(asvab_vars)'

# Perform PCA to reduce ASVAB variables to the first principal component
M = fit(PCA, asvabMat; maxoutdim=1)
asvabPCA = MultivariateStats.transform(M, asvabMat)

# Reshape the PCA result to be N-length (one observation per row)
asvabPCA = reshape(asvabPCA, size(asvabPCA, 2))
df.PCA1 = asvabPCA  # Add PCA1 as a new column in `df`

# Define and fit the regression model with the first principal component of ASVAB variables
pca_formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + PCA1)
pca_model = lm(pca_formula, df)

# Display the regression results
println("Regression Results with First Principal Component of ASVAB Variables:")
display(coeftable(pca_model))


# Question 5
F = fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabFactor = MultivariateStats.transform(F, asvabMat)

# Reshape the factor result to an N-length array to match `df`
asvabFactor = reshape(asvabFactor, size(asvabFactor, 2))
df.Factor1 = asvabFactor  # Add the first factor as a new column in `df`

# Define and fit the regression model with the first factor as a covariate
factor_formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + Factor1)
factor_model = lm(factor_formula, df)

# Display the regression results
println("Regression Results with First Factor of ASVAB Variables:")
display(coeftable(factor_model))



# Question 6
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
response = HTTP.get(url)
data = CSV.read(IOBuffer(response.body), DataFrame)

# 2. Define variables
log_wage = data[!, :logwage]
black = data[!, :black]
hispanic = data[!, :hispanic]
female = data[!, :female]
school = data[!, :schoolt]
gradHS = data[!, :gradHS]
grad4yr = data[!, :grad4yr]
asvab = Matrix(data[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])

# 3. Define log-likelihood function
function log_likelihood(params)
    α0 = params[1:6]
    α1 = params[7:12]
    α2 = params[13:18]
    α3 = params[19:24]
    γ = params[25:30]
    β = params[31:37]
    δ = params[38]
    σ_asvab = abs.(params[39:44])  # Ensure non-negative variances
    σ_wage = abs(params[45])

    log_likelihood = 0.0

    # Use lgwt to get Gauss-Legendre points and weights
    ξ_points, ξ_weights = lgwt(40, -1, 1)  # Increase points for better accuracy
    
    # Iterate over each observation
    for i in 1:size(data, 1)
        xi = [1, black[i], hispanic[i], female[i]]
        asvab_i = asvab[i, :]
        
        wage_pred = β[1] + β[2] * black[i] + β[3] * hispanic[i] + β[4] * female[i] +
                    β[5] * school[i] + β[6] * gradHS[i] + β[7] * grad4yr[i]
        
        integral = 0.0
        for (ξ, w) in zip(ξ_points, ξ_weights)
            asvab_resid = asvab_i .- (α0 .+ α1 * black[i] .+ α2 * hispanic[i] .+ α3 * female[i] .+ γ * ξ)
            wage_resid = log_wage[i] - (wage_pred + δ * ξ)
            asvab_likelihood = prod(pdf(Normal(0, σ_asvab[j]), asvab_resid[j]) for j in 1:6)
            wage_likelihood = pdf(Normal(0, σ_wage), wage_resid)

            integral += asvab_likelihood * wage_likelihood * w
        end
        
        # Avoid log(0) by adding a small value if integral is zero
        if integral > 0
            log_likelihood += log(integral)
        else
            log_likelihood += -1e10  # Large negative penalty
        end
        
        # Debugging output
        # println("Observation: ", i, ", Integral: ", integral)
    end

    return -log_likelihood
end

# Optimization with initial parameter setup
initial_params = vcat(zeros(38), fill(0.1, 7))  # Start with zeros for most, small positive values for variances
result = optimize(log_likelihood, initial_params, BFGS())  # Use BFGS instead of Nelder-Mead
println(result)


# Question 7
using Test, HTTP, CSV, DataFrames, Optim, LinearAlgebra, Random, Statistics, Distributions, MultivariateStats, StatsModels, GLM

# Test 1: HTTP request and CSV read
@testset "HTTP Request and CSV Read" begin
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
    response = HTTP.get(url)
    @test response.status == 200
    df = CSV.read(IOBuffer(response.body), DataFrame)
    @test !isempty(df)
end

# Test 2: Linear Regression Model
@testset "Linear Regression Model" begin
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr)
    model = lm(formula, df)
    @test !isempty(coeftable(model))  # Check if coefficients are computed
end

# Test 3: Correlation Matrix
@testset "Correlation Matrix" begin
    asvab_vars = select(df, r"asvab")
    cor_matrix = cor(Matrix(asvab_vars))
    @test size(cor_matrix) == (6, 6)  # Check dimensions
    @test all(-1 .<= cor_matrix .<= 1)  # Values should be between -1 and 1
end

# Test 4: PCA
@testset "PCA Reduction" begin
    asvabMat = Matrix(select(df, r"asvab"))'
    M = fit(PCA, asvabMat; maxoutdim=1)
    asvabPCA = MultivariateStats.transform(M, asvabMat)
    @test size(asvabPCA, 1) == 1  # Should have one principal component
end

# Test 5: Factor Analysis
@testset "Factor Analysis" begin
    F = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    asvabFactor = MultivariateStats.transform(F, asvabMat)
    @test size(asvabFactor, 1) == 1  # Should have one factor loading
end

# Test 6: Log-likelihood Function Optimization
@testset "Log-likelihood Optimization" begin
    function log_likelihood_test(params)
        # Simplified log-likelihood function for testing
        return sum(params .^ 2)  # Dummy function for optimization test
    end
    initial_params = fill(0.1, 45)
    result = optimize(log_likelihood_test, initial_params, BFGS())
    @test isfinite(result.minimum) 
end
