using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using Test

include("create_grids.jl")

# Step 1: Data loading and preprocessing
function load_and_preprocess_data(url::String)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df, 1))  # Add bus_id to identify individual buses
    return df
end

function reshape_data(df::DataFrame)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)  # Rename the value column to Y (the outcome variable)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfy_long, Not(:variable))

    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)  # Rename the value column to Odometer (independent variable)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])

    sort!(df_long, [:bus_id, :time])

    return df_long
end

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = load_and_preprocess_data(url)
df_long = reshape_data(df)

# Step 2: Estimate the flexible logit model
df_long.Branded = convert(Vector{Int64}, df_long.Branded)
df_long.Y = convert(Vector{Int64}, df_long.Y)

logit_model = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time),
                    df_long, 
                    Binomial(), 
                    LogitLink())

println(coeftable(logit_model))

# Step 3(a): Construct the state transition matrices
function construct_transition_matrices()
    zval, zbin, xval, xbin, xtran = create_grids()
    return zval, zbin, xval, xbin, xtran
end

zval, zbin, xval, xbin, xtran = construct_transition_matrices()

# Step 3(b): Calculate future value terms (FVT1)
function calculate_FVT1(df_long::DataFrame, FV::Array{Float64, 3}, xtran::Matrix{Float64}, xbin::Int64, zbin::Int64)
    FVT1 = zeros(size(df_long, 1), size(FV, 3) - 1)

    for i in 1:size(df_long, 1)
        for t in 1:(size(FV, 3) - 1)
            row1 = findfirst(x -> x >= df_long.Odometer[i], xtran[:, 1])
            if row1 === nothing
                row1 = argmin(abs.(xtran[:, 1] .- df_long.Odometer[i]))
            end
            row0 = Int(df_long.Branded[i]) + 1

            if row0 > 0 && row1 > 0
                FVT1[i, t] = sum((xtran[row1, :] .- xtran[row0, :]) * FV[row1, row0, t+1])
            end
        end
    end

    return FVT1
end

# Future value array FV initialization
FV = zeros(size(xtran, 1), 2, 21)  # Assuming T=20, initialize FV with size (grid points, 2 branded states, 21 time periods)
β = 0.9  # Discount factor

# Create the state dataframe
state_df = DataFrame(Odometer = kron(ones(zbin), xval),
                        RouteUsage = kron(ones(xbin), zval),
                        Branded = zeros(length(kron(ones(zbin), xval))),
                        time = zeros(length(kron(ones(zbin), xval))))

# Loop over time periods and brand states to calculate FV
for t in 2:20
    for b in 0:1
        state_df.Branded .= b
        state_df.time .= t

        # Predict probabilities (p0)
        p0 = predict(logit_model, state_df)

        # Update FV with the calculated future value term
        FV[:, b+1, t+1] .= -β * log.(p0)
    end
end

FVT1 = calculate_FVT1(df_long, FV, xtran, xbin, zbin)

# Step 3(c): Estimate the structural parameters
# Ensure that FVT1[:, 1]' is reshaped into a vector that can be added as a column
df_long = @transform(df_long, :fv = vec(FVT1[:, 1])) # Add the first period of FVT1 as a column

theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset = df_long.fv)

println(coeftable(theta_hat_ccp_glm))



function full_process()
    # Step 1: Data loading and preprocessing
    function load_and_preprocess_data(url::String)
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = @transform(df, :bus_id = 1:size(df, 1))  # Add bus_id to identify individual buses
        return df
    end

    function reshape_data(df::DataFrame)
        dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
        dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
        rename!(dfy_long, :value => :Y)  # Rename the value column to Y (the outcome variable)
        dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df, 1))))
        select!(dfy_long, Not(:variable))

        dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
        dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
        rename!(dfx_long, :value => :Odometer)  # Rename the value column to Odometer (independent variable)
        dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df, 1))))
        select!(dfx_long, Not(:variable))

        df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])

        sort!(df_long, [:bus_id, :time])

        return df_long
    end

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = load_and_preprocess_data(url)
    df_long = reshape_data(df)

    # Step 2: Estimate the flexible logit model
    df_long.Branded = convert(Vector{Int64}, df_long.Branded)
    df_long.Y = convert(Vector{Int64}, df_long.Y)

    logit_model = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time),
                      df_long, 
                      Binomial(), 
                      LogitLink())
    
    println(coeftable(logit_model))

    # Step 3(a): Construct the state transition matrices
    function construct_transition_matrices()
        zval, zbin, xval, xbin, xtran = create_grids()
        return zval, zbin, xval, xbin, xtran
    end

    zval, zbin, xval, xbin, xtran = construct_transition_matrices()

    # Step 3(b): Calculate future value terms (FVT1)
    function calculate_FVT1(df_long::DataFrame, FV::Array{Float64, 3}, xtran::Matrix{Float64}, xbin::Int64, zbin::Int64)
        FVT1 = zeros(size(df_long, 1), size(FV, 3) - 1)

        for i in 1:size(df_long, 1)
            for t in 1:(size(FV, 3) - 1)
                row1 = findfirst(x -> x >= df_long.Odometer[i], xtran[:, 1])
                if row1 === nothing
                    row1 = argmin(abs.(xtran[:, 1] .- df_long.Odometer[i]))
                end
                row0 = Int(df_long.Branded[i]) + 1

                if row0 > 0 && row1 > 0
                    FVT1[i, t] = sum((xtran[row1, :] .- xtran[row0, :]) * FV[row1, row0, t+1])
                end
            end
        end

        return FVT1
    end

    # Future value array FV initialization
    FV = zeros(size(xtran, 1), 2, 21)  # Assuming T=20, initialize FV with size (grid points, 2 branded states, 21 time periods)
    β = 0.9  # Discount factor

    # Create the state dataframe
    state_df = DataFrame(Odometer = kron(ones(zbin), xval),
                         RouteUsage = kron(ones(xbin), zval),
                         Branded = zeros(length(kron(ones(zbin), xval))),
                         time = zeros(length(kron(ones(zbin), xval))))

    # Loop over time periods and brand states to calculate FV
    for t in 2:20
        for b in 0:1
            state_df.Branded .= b
            state_df.time .= t

            # Predict probabilities (p0)
            p0 = predict(logit_model, state_df)

            # Update FV with the calculated future value term
            FV[:, b+1, t+1] .= -β * log.(p0)
        end
    end

    FVT1 = calculate_FVT1(df_long, FV, xtran, xbin, zbin)

    # Step 3(c): Estimate the structural parameters
    # Ensure that FVT1[:, 1]' is reshaped into a vector that can be added as a column
    df_long = @transform(df_long, :fv = vec(FVT1[:, 1])) # Add the first period of FVT1 as a column

    theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset = df_long.fv)

    println(coeftable(theta_hat_ccp_glm))

end

# Use @time to measure the execution time of the full process
@time full_process()


function simple_test_full_process()

    # Test data loading
    @testset "Simple Test Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
        df = load_and_preprocess_data(url)
        
        # Simple test: Check if the DataFrame is not empty
        @test !isempty(df)
    end

    # Test data reshaping
    @testset "Simple Test Data Reshaping" begin
        # Load the data again to define df
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
        df = load_and_preprocess_data(url)

        df_long = reshape_data(df)
        
        # Simple test: Check if the reshaped DataFrame is not empty
        @test !isempty(df_long)
    end

    # Test model estimation
    @testset "Simple Test Logit Model" begin
        logit_model = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time),
                          df_long, 
                          Binomial(), 
                          LogitLink())
        
        # Simple test: Check if the model coefficients table is not empty
        coef_table = coeftable(logit_model)
        @test !isempty(coef_table)
    end

end


# Call the simple test function
simple_test_full_process()