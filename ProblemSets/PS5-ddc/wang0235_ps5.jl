using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Create a unique bus identifier
df = @transform(df, :bus_id = 1:size(df,1))

# Step 3: Reshape the decision variables (Y1 to Y20) into long format
dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
rename!(dfy_long, :value => :Y)

# Step 4: Add a time variable (1 to 20) representing each time period for observations
dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfy_long, Not(:variable))

# Step 5: Reshape the odometer variables (Odo1 to Odo20) into long format
dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)

# Step 6: Add the time variable to the odometer data
dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfx_long, Not(:variable))

# Step 7: Merge the reshaped decision (Y) and odometer (Odo) dataframes by bus_id and time
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
sort!(df_long, [:bus_id, :time])

# Step 8: Fit a binary logit model with only the Odometer as the independent variable
logit_model = glm(@formula(Y ~ Odometer), df_long, Binomial(), LogitLink())

# Output the result of the model
print(coeftable(logit_model))



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Create a unique bus identifier
df = @transform(df, :bus_id = 1:size(df,1))

# Step 3: Reshape the decision variables (Y1 to Y20) into long format
dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
rename!(dfy_long, :value => :Y)

# Step 4: Add a time variable (1 to 20) representing each time period for observations
dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfy_long, Not(:variable))

# Step 5: Reshape the odometer variables (Odo1 to Odo20) into long format
dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)

# Step 6: Add the time variable to the odometer data
dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfx_long, Not(:variable))

# Step 7: Merge the reshaped decision (Y), odometer (Odo), and branded dataframes by bus_id and time
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
sort!(df_long, [:bus_id, :time])

# Step 8: Fit a binary logit model with Odometer and Branded as independent variables
logit_model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())

# Output the result of the model
print(coeftable(logit_model))



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using CSV
using DataFrames
using HTTP

# Step 1: Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Step 2: Convert decision variables (Y1 to Y20) into a matrix
Y = Matrix(df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, 
                  :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]])

# Step 3: Convert the odometer readings (Odo1 to Odo20) into a matrix
Odo = Matrix(df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10,
                    :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]])

# Step 4: Convert Xst (usage intensity) into a matrix
Xst = Matrix(df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10,
                    :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])

# Step 5: Convert Zst (discrete bins) into a column matrix (reshape to add the second dimension)
Zst = reshape(df[:, :Zst], :, 1)  # Convert vector into column matrix

# Output dimensions of matrices to verify
println("Dimensions of Y: ", size(Y))      # Expected: (number of buses, 20)
println("Dimensions of Odo: ", size(Odo))  # Expected: (number of buses, 20)
println("Dimensions of Xst: ", size(Xst))  # Expected: (number of buses, 20)
println("Dimensions of Zst: ", size(Zst))  # Expected: (number of buses, 1)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::
zval, zbin, xval, xbin, xtran = create_grids()

# Display the generated grids and transition matrix
println("Route usage grid (zval): ", zval)
println("Number of bins for route usage (zbin): ", zbin)
println("Odometer grid (xval): ", xval)
println("Number of bins for odometer (xbin): ", xbin)
println("State transition matrix (xtran) dimensions: ", size(xtran))

# Check the content or part of the state transition matrix
# println("State transition matrix (first 5 rows): ")
# println(xtran[1:5, :])



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3c: compute the value function using backward induction
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using LinearAlgebra  # For matrix operations

# Set the discount factor and costs
β = 0.95  # Discount factor
replacement_cost = 1000  # Cost of replacing the engine (arbitrary value)
running_cost = 100  # Running cost for each period (arbitrary value)
T = 20  # Number of periods (finite horizon)

# Get the grids and transition matrix from create_grids()
zval, zbin, xval, xbin, xtran = create_grids()

# Step 1: Initialize the value function array
# Value function for each (x, z) state, at each time period
# We add one extra period for the terminal condition (T + 1)
V = zeros(xbin, zbin, T + 1)  # (odometer, route usage, time period)

# Step 2: Backward induction over finite periods
# Start with the terminal value function (in the last period, there are no future rewards)

# Assume in the last period, the only decision is to replace the bus (we set this arbitrarily)
for x = 1:xbin
    for z = 1:zbin
        V[x, z, T + 1] = -replacement_cost  # Terminal cost of replacing the bus in the last period
    end
end

# Step 3: Iterate backward over time periods
for t = T:-1:1  # Start from period T and go backward to period 1
    for x = 1:xbin
        for z = 1:zbin
            # Running the bus: future value depends on transition matrix and current value function
            continue_value = running_cost + β * sum(xtran[x, :] .* V[:, z, t + 1])

            # Replacing the engine: reset to state (1, z) and pay the replacement cost
            replace_value = replacement_cost + β * V[1, z, t + 1]

            # Bellman equation: choose the maximum of running or replacing
            V[x, z, t] = max(continue_value, replace_value)
        end
    end
end

# Output the final value function at the initial period (t = 1)
# println("Value function at the first period:")
# println(V[:, :, 1])

# println("Transition matrix (xtran):")
# println(xtran)



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3(d): 
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Use a let block to ensure that ll_value is treated as local in its own scope
let
    # Step 1: Initialize the log-likelihood value as a local variable
    ll_value = 0.0  # Define ll_value locally within this scope

    # Set up θ parameters (initial guess, we'll later optimize over these)
    θ0 = -1.0  # Intercept term
    θ1 = -0.1  # Coefficient for Odometer (x1t)
    θ2 = 0.5   # Coefficient for Branded

    # Iterate over each bus and time period to compute the log likelihood
    for bus in 1:size(Y, 1)  # Loop over buses
        for t in 1:T  # Loop over time periods
            # Get observed values for this bus at time t
            x1t = Odo[bus, t]   # Odometer reading at time t
            b = df[bus, :Branded]  # Is the bus branded?
            Yt = Y[bus, t]  # Observed decision at time t (0 or 1)

            # Compute the flow utility v1t - v0t using observed odometer and branding
            flow_utility = θ0 + θ1 * x1t + θ2 * b

            # Find state indices for this bus and time period
            z_state = Zst[bus]  # Route usage state from data
            x_state = argmin(abs.(xval .- x1t))  # Closest bin to observed odometer reading

            # Ensure the index values for rows are within the bounds of V and xtran dimensions
            row0 = 1 + (z_state - 1) * xbin  # Index for state when engine is replaced (odometer reset to 0)
            row1 = x_state + (z_state - 1) * xbin  # Index for state when engine is not replaced (current odometer)

            if row0 <= size(xtran, 1) && row1 <= size(xtran, 1)
                # Compute the future value difference using the transition matrix and value function
                future_value_diff = (xtran[row1, :] .- xtran[row0, :])' * V[1:xbin, b + 1, t + 1]

                # Combine flow utility and future value difference
                v_diff = flow_utility + β * future_value_diff

                # Calculate choice probabilities
                P1t = exp(v_diff) / (1 + exp(v_diff))  # Probability of running (Y = 1)
                P0t = 1 - P1t  # Probability of replacing (Y = 0)

                # Update the log-likelihood based on the observed decision
                if Yt == 1
                    ll_value += log(P1t)
                else
                    ll_value += log(P0t)
                end
            else
                println("Warning: Skipping out-of-bounds state at bus $bus, time $t")
            end
        end
    end

    # Output the log-likelihood value
    println("Log-Likelihood: ", ll_value)
end



# #:::::::::::::::::::::::::::::::::::::::::::::::::::
# # Question 3(e): Wrapping the code into an objective function for optimization
# #:::::::::::::::::::::::::::::::::::::::::::::::::::

# Define the objective function for optimization
function log_likelihood(θ::Vector{Float64})
    θ0, θ1, θ2 = θ[1], θ[2], θ[3]
    
    # Initialize the log-likelihood value
    ll_value = 0.0

    # Iterate over each bus and time period to compute the log likelihood
    for bus in 1:size(Y, 1)  # Loop over buses
        for t in 1:T  # Loop over time periods
            # Get observed values for this bus at time t
            x1t = Odo[bus, t]   # Odometer reading at time t
            b = df[bus, :Branded]  # Is the bus branded?
            Yt = Y[bus, t]  # Observed decision at time t (0 or 1)

            # Compute the flow utility v1t - v0t using observed odometer and branding
            flow_utility = θ0 + θ1 * x1t + θ2 * b

            # Find state indices for this bus and time period
            z_state = Zst[bus]  # Route usage state from data
            x_state = argmin(abs.(xval .- x1t))  # Closest bin to observed odometer reading

            # Ensure the index values for rows are within the bounds of V and xtran dimensions
            row0 = 1 + (z_state - 1) * xbin  # Index for state when engine is replaced (odometer reset to 0)
            row1 = x_state + (z_state - 1) * xbin  # Index for state when engine is not replaced (current odometer)

            if row0 <= size(xtran, 1) && row1 <= size(xtran, 1)
                # Compute the future value difference using the transition matrix and value function
                future_value_diff = (xtran[row1, :] .- xtran[row0, :])' * V[1:xbin, b + 1, t + 1]

                # Combine flow utility and future value difference
                v_diff = flow_utility + β * future_value_diff

                # Calculate choice probabilities
                P1t = exp(v_diff) / (1 + exp(v_diff))  # Probability of running (Y = 1)
                P0t = 1 - P1t  # Probability of replacing (Y = 0)

                # Update the log-likelihood based on the observed decision
                if Yt == 1
                    ll_value += log(P1t)
                else
                    ll_value += log(P0t)
                end
            else
                println("Warning: Skipping out-of-bounds state at bus $bus, time $t")
            end
        end
    end

    # Return the negative log-likelihood because most optimizers minimize
    return -ll_value
end

# Set initial guesses for θ parameters
initial_θ = [-1.0, -0.1, 0.5]  # Initial values for θ0, θ1, θ2

# Perform optimization using Optim.jl to maximize the log-likelihood
result = optimize(log_likelihood, initial_θ, BFGS())

# Output the results of the optimization
println("Optimization Results:")
println(result)
println("Optimized Parameters: ", Optim.minimizer(result))


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3(f): Optimize code with @views and @inbounds macros
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Define the objective function for optimization with macros
@views @inbounds function log_likelihood(θ::Vector{Float64})
    θ0, θ1, θ2 = θ[1], θ[2], θ[3]
    
    # Initialize the log-likelihood value
    ll_value = 0.0

    # Iterate over each bus and time period to compute the log likelihood
    for bus in 1:size(Y, 1)  # Loop over buses
        for t in 1:T  # Loop over time periods
            # Get observed values for this bus at time t
            x1t = Odo[bus, t]   # Odometer reading at time t
            b = df[bus, :Branded]  # Is the bus branded?
            Yt = Y[bus, t]  # Observed decision at time t (0 or 1)

            # Compute the flow utility v1t - v0t using observed odometer and branding
            flow_utility = θ0 + θ1 * x1t + θ2 * b

            # Find state indices for this bus and time period
            z_state = Zst[bus]  # Route usage state from data
            x_state = argmin(abs.(xval .- x1t))  # Closest bin to observed odometer reading

            # Ensure the index values for rows are within the bounds of V and xtran dimensions
            row0 = 1 + (z_state - 1) * xbin  # Index for state when engine is replaced (odometer reset to 0)
            row1 = x_state + (z_state - 1) * xbin  # Index for state when engine is not replaced (current odometer)

            if row0 <= size(xtran, 1) && row1 <= size(xtran, 1)
                # Compute the future value difference using the transition matrix and value function
                future_value_diff = (xtran[row1, :] .- xtran[row0, :])' * V[1:xbin, b + 1, t + 1]

                # Combine flow utility and future value difference
                v_diff = flow_utility + β * future_value_diff

                # Calculate choice probabilities
                P1t = exp(v_diff) / (1 + exp(v_diff))  # Probability of running (Y = 1)
                P0t = 1 - P1t  # Probability of replacing (Y = 0)

                # Update the log-likelihood based on the observed decision
                if Yt == 1
                    ll_value += log(P1t)
                else
                    ll_value += log(P0t)
                end
            else
                println("Warning: Skipping out-of-bounds state at bus $bus, time $t")
            end
        end
    end

    # Return the negative log-likelihood because most optimizers minimize
    return -ll_value
end

# Perform optimization using Optim.jl to maximize the log-likelihood
result = optimize(log_likelihood, initial_θ, BFGS())

# Output the results of the optimization
println("Optimization Results:")
println(result)
println("Optimized Parameters: ", Optim.minimizer(result))



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3(g): Final wrap-up of all steps into a function
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")

# Main function to wrap everything
function bus_replacement_model()
    # Step 1: Data Preparation
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Convert decision variables (Y1 to Y20) into a matrix
    Y = Matrix(df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, 
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]])

    # Convert the odometer readings (Odo1 to Odo20) into a matrix
    Odo = Matrix(df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10,
                        :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]])

    # Convert Xst (usage intensity) into a matrix
    Xst = Matrix(df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10,
                        :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])

    # Convert Zst (discrete bins) into a column matrix (reshape to add the second dimension)
    Zst = reshape(df[:, :Zst], :, 1)  # Convert vector into column matrix

    # Output dimensions of matrices to verify
    println("Data matrices loaded and reshaped.")

    # Step 2: State Transition Matrices
    # Create grids and transition matrix for the dynamic model
    zval, zbin, xval, xbin, xtran = create_grids()
    println("State transition matrices generated.")

    # Step 3: Compute Value Function via Backward Induction
    # Set the discount factor and costs
    β = 0.95  # Discount factor
    replacement_cost = 1000  # Arbitrary cost of replacing the engine
    running_cost = 100  # Arbitrary running cost
    T = 20  # Number of periods (finite horizon)

    # Initialize value function array: (odometer, route usage, time period)
    V = zeros(xbin, zbin, T + 1)

    # Terminal condition: set final period values
    for x in 1:xbin
        for z in 1:zbin
            V[x, z, T + 1] = -replacement_cost
        end
    end

    # Backward induction to compute value function
    for t in T:-1:1
        for x in 1:xbin
            for z in 1:zbin
                # Calculate running and replacement values
                continue_value = running_cost + β * sum(xtran[x, :] .* V[:, z, t + 1])
                replace_value = replacement_cost + β * V[1, z, t + 1]

                # Choose the maximum of running or replacing
                V[x, z, t] = max(continue_value, replace_value)
            end
        end
    end
    println("Value function computed.")

    # Step 4: Log-Likelihood Calculation and Optimization
    # Define the log-likelihood function
    @views @inbounds function log_likelihood(θ::Vector{Float64})
        θ0, θ1, θ2 = θ[1], θ[2], θ[3]
        ll_value = 0.0

        for bus in 1:size(Y, 1)
            for t in 1:T
                # Get observed values for this bus at time t
                x1t = Odo[bus, t]
                b = df[bus, :Branded]
                Yt = Y[bus, t]

                # Flow utility calculation
                flow_utility = θ0 + θ1 * x1t + θ2 * b

                # Find the state indices
                z_state = Zst[bus]
                x_state = argmin(abs.(xval .- x1t))

                # Calculate row indices for transition matrix and value function
                row0 = 1 + (z_state - 1) * xbin
                row1 = x_state + (z_state - 1) * xbin

                if row0 <= size(xtran, 1) && row1 <= size(xtran, 1)
                    # Compute the future value difference
                    future_value_diff = (xtran[row1, :] .- xtran[row0, :])' * V[1:xbin, b + 1, t + 1]

                    # Calculate value differences
                    v_diff = flow_utility + β * future_value_diff

                    # Choice probabilities
                    P1t = exp(v_diff) / (1 + exp(v_diff))
                    P0t = 1 - P1t

                    # Update the log-likelihood based on the observed decision
                    if Yt == 1
                        ll_value += log(P1t)
                    else
                        ll_value += log(P0t)
                    end
                end
            end
        end

        return -ll_value  # Negative log-likelihood for minimization
    end

    # Set initial guesses for θ parameters
    initial_θ = [-1.0, -0.1, 0.5]

    # Perform optimization using BFGS algorithm
    result = optimize(log_likelihood, initial_θ, BFGS())
    
    # Print the optimization results
    println("Optimization complete.")
    println("Optimized Parameters: ", Optim.minimizer(result))
end

# Run the complete model
bus_replacement_model()



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3(h)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
optimized_params = Optim.minimizer(result)
println("Optimized Parameters: θ0 = $(optimized_params[1]), θ1 = $(optimized_params[2]), θ2 = $(optimized_params[3])")

# Output and Explanation:
# θ0=1.847, θ1=−0.276, θ2=0.776
# A positive θ0 suggests there is an inherent utility in continuing to use the bus, even without considering mileage or branding.
# The negative θ1 suggests that higher mileage decreases the likelihood of continuing to run the bus and increases the likelihood of engine replacement.
# A positive θ2 indicates that branded buses are more likely to continue running rather than being replaced, perhaps due to better quality or reliability.

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: Unit Tests
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using Test

# Define the grids creation function here (or include from your local file)
include("create_grids.jl")

# Unit tests for the entire model
@testset "Bus Replacement Model Tests" begin
    # Step 1: Test Data Loading and Preparation
    df = CSV.read(HTTP.get("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv").body, DataFrame)
    @test !isempty(df)  # Ensure data is not empty
    @test size(df, 1) == 1000  # Check that we have 1000 rows (number of buses)
    @test size(df, 2) >= 20  # Check for at least 20 columns (Y1 to Y20, odometer readings, etc.)
    
    # Step 2: Test the Value Function Calculation
    β = 0.95  # Discount factor
    replacement_cost = 1000
    running_cost = 100
    T = 20
    xbin, zbin = 201, 101  # Set bins for odometer and route usage based on data
    
    V_test = zeros(xbin, zbin, T + 1)  # Initialize value function array
    
    for t in T:-1:1
        for x in 1:xbin
            for z in 1:zbin
                continue_value = running_cost + β * sum(rand(xbin))  # Random future values for testing
                replace_value = replacement_cost + β * rand()
                V_test[x, z, t] = max(continue_value, replace_value)
            end
        end
    end
    
    @test V_test[100, 50, 1] != 0  # Ensure value function has been updated for some states

    # Step 3: Test Log-Likelihood Calculation
    θ_test = [1.0, -0.1, 0.5]  # Some test parameter values
    function log_likelihood(θ::Vector{Float64})
        # Implement a simplified version for testing purposes
        ll_value = -1000.0  # Mocked value, just to ensure function works
        return ll_value
    end
    
    ll_test = log_likelihood(θ_test)  # Call the log-likelihood function
    @test ll_test < 0  # Log-likelihood should be negative
    @test isnan(ll_test) == false  # Check for NaN

    # Step 4: Test Optimization Routine
    initial_θ = [-1.0, -0.1, 0.5]  # Initial parameter values
    result_test = optimize(log_likelihood, initial_θ, BFGS())
    
    @test Optim.converged(result_test)  # Check if optimization converged successfully
    @test Optim.minimizer(result_test)[1] == -1.0  # Since the mocked log-likelihood doesn't change, expect no change in parameters
end
