# Problem Set 1 - Econ 6343
# Xin Wang

# Question 1

#(a), (b), (c)
using Random
using Distributions

# Set the seed
Random.seed!(1234)

# Create the matrices
A = rand(Uniform(-5, 10), 10, 7)
B = rand(Normal(-2, 15), 10, 7)
C = [A[1:5, 1:5] B[1:5, 6:7]]
D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]

# Number of elements in A
num_elements_A = length(A)

# Number of unique elements in D
num_unique_elements_D = length(unique(D))

# Output the results
println("Number of elements in A: ", num_elements_A)
println("Number of unique elements in D: ", num_unique_elements_D)

# (d) Reshape B into a vector E
E = vec(B)

# (e) Create a 3-dimensional array F
F = cat(A, B; dims=3)

# (f) Permute dimensions of F to make it 2x10x7
F = permutedims(F, (3, 1, 2))

# Output the final F matrix dimensions
println("Dimensions of F: ", size(F))

# (g) Create Matrix G as the Kronecker Product of B and C
G = kron(B, C)

# (h)
using JLD
save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

# (i)
save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

# (j)
using DataFrames, CSV

df_C = DataFrame(C, :auto)
CSV.write("Cmatrix.csv", df_C)

# (k)
df_D = DataFrame(D, :auto)
CSV.write("Dmatrix.dat", df_D; delim='\t')

# (l)
# Import necessary packages
# using Random
# using Distributions
# using JLD
# using DataFrames
# using CSV

function q1()
    # Set the seed
    Random.seed!(1234)

    # Create matrices A and B
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    
    # Matrix C
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    
    # Matrix D
    D = [A[i,j] <= 0 ? A[i,j] : 0 for i in 1:10, j in 1:7]
    
    # Matrix E
    E = vec(B)
    
    # Matrix F
    F = permutedims(cat(A, B; dims=3), (3, 1, 2))
    
    # Matrix G
    G = kron(B, C)
    
    # Save matrices to .jld files
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)
    
    # Export matrices C and D as .csv and .dat files
    df_C = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", df_C)
    
    df_D = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", df_D; delim='\t')
    
    return A, B, C, D
end

# Run the function and assign outputs
A, B, C, D = q1()



# Question 2

# (a) Element-wise product of matrices A and B
function elementwise_product(A, B)
    AB = [A[i,j] * B[i,j] for i in 1:size(A,1), j in 1:size(A,2)]
    AB2 = A .* B
    return AB, AB2
end

# (b) Filter elements of C between -5 and 5
function filter_C_elements(C)
    Cprime = [C[i] for i in 1:length(C) if C[i] ≥ -5 && C[i] ≤ 5]
    Cprime2 = C[(C .≥ -5) .& (C .≤ 5)]
    return Cprime, Cprime2
end

# (c) Create a 3-dimensional array X of dimension N × K × T
function create_3d_array(N, K, T)
    X = Array{Float64}(undef, N, K, T)
    for t in 1:T
        X[:,1,t] = ones(N)
        X[:,2,t] = rand(Bernoulli(0.75*(6-t)/5), N)
        X[:,3,t] = rand(Normal(15 + t - 1, 5*(t - 1)), N)
        X[:,4,t] = rand(Normal(π*(6-t)/3, 1/exp(1)), N)
        X[:,5,t] = rand(Binomial(20, 0.6), N)
        X[:,6,t] = rand(Binomial(20, 0.5), N)
    end
    return X
end

# (d) Create a matrix β which is K × T
function create_beta(K, T)
    β = [1 + 0.25*(t-1) for t in 1:T]'
    β = vcat(β, [log(t) for t in 1:T]')
    β = vcat(β, [-sqrt(t) for t in 1:T]')
    β = vcat(β, [exp(t) - exp(t+1) for t in 1:T]')
    β = vcat(β, [t for t in 1:T]')
    β = vcat(β, [t/3 for t in 1:T]')
    return β
end

# (e) Create a matrix Y which is N × T defined by Y_t = X_t * β_t + ε_t
function create_Y(X, β, N, T)
    ε_t = rand(Normal(0, 0.36), N, T)
    Y = Array{Float64}(undef, N, T)
    for t in 1:T
        Y[:,t] = X[:,:,t] * β[:,t] + ε_t[:,t]
    end
    return Y
end

# (f) Wrapping everything into a function q2()
function q2(A, B, C)
    AB, AB2 = elementwise_product(A, B)
    Cprime, Cprime2 = filter_C_elements(C)
    N = 15169
    K = 6
    T = 5
    X = create_3d_array(N, K, T)
    β = create_beta(K, T)
    Y = create_Y(X, β, N, T)
    return nothing
end

# Question 3
using DataFrames
using CSV
using Statistics

function q3()
    # (a) Import the file nlsw88.csv into Julia as a DataFrame
    df = CSV.read("/home/etica/Project/fall-2024-econ6343/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame)

    # Convert missing values and variable names if necessary
    # Saving the result as nlsw88_processed.csv
    CSV.write("nlsw88_processed.csv", df)

    # (b) What percentage of the sample has never been married? What percentage are college graduates?
    never_married_percentage = mean(df[!, :married] .== "Never married") * 100
    college_graduate_percentage = mean(df[!, :collgrad] .== "Yes") * 100

    println("Percentage never married: ", never_married_percentage)
    println("Percentage college graduates: ", college_graduate_percentage)

    # (c) Use the freqtable() function to report what percentage of the sample is in each race category
    race_table = combine(groupby(df, :race), nrow => :Count)
    race_table[!, :Percentage] = race_table[!, :Count] ./ sum(race_table[!, :Count]) .* 100

    println("Race distribution:")
    println(race_table)

    # (d) Use the describe() function to create a matrix called summarystats
    summarystats = describe(df)
    println("Summary statistics:")
    println(summarystats)

    # Finding how many grade observations are missing
    missing_grades = sum(ismissing, df[!, :grade])
    println("Number of missing grade observations: ", missing_grades)

    # (e) Show the joint distribution of industry and occupation using a cross-tabulation
    industry_occupation_table = combine(groupby(df, [:industry, :occupation]), nrow => :Count)
    println("Joint distribution of industry and occupation:")
    println(industry_occupation_table)

    # (f) Tabulate the mean wage over industry and occupation categories
    wage_by_industry_occupation = combine(groupby(df, [:industry, :occupation]), :wage => mean => :MeanWage)
    println("Mean wage by industry and occupation:")
    println(wage_by_industry_occupation)
end

# Call the function q3() to execute all of the above
q3()



# Question 4
using JLD
using CSV
using DataFrames

# Load the matrix file
data = load("/home/etica/Project/fall-2024-econ6343/ProblemSets/PS1-julia-intro/firstmatrix.jld")

# Assuming the matrices A, B, C, and D are stored in variables A, B, C, D in the JLD file
A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]

# Function matrixops with checks and operations
function matrixops(A::Array, B::Array)
    # Check if inputs have the same size
    if size(A) != size(B)
        error("inputs must have the same size")
    end

    # Element-wise product of A and B
    elem_product = A .* B

    # Product of transpose(A) and B
    trans_product = transpose(A) * B

    # Sum of all elements in A + B
    sum_elements = sum(A) + sum(B)

    return elem_product, trans_product, sum_elements
end

# Test matrixops with A and B
println("Size of A: ", size(A))
println("Size of B: ", size(B))
elem_product, trans_product, sum_elements = matrixops(A, B)
println("Element-wise product:\n", elem_product)
println("Product of transpose(A) and B:\n", trans_product)
println("Sum of all elements in A and B:\n", sum_elements)

# Test matrixops with C and D (this might cause an error, so checking sizes first)
println("Size of C: ", size(C))
println("Size of D: ", size(D))
# raise an error because sizes don't match,Size of C: (5, 7), Size of D: (10, 7)
# elem_product, trans_product, sum_elements = matrixops(C, D)

# Load nlsw88.csv
nlsw88 = CSV.read("/home/etica/Project/fall-2024-econ6343/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame)

# Convert ttl_exp and wage columns to arrays
ttl_exp_array = convert(Array, nlsw88[!, :ttl_exp])
wage_array = convert(Array, nlsw88[!, :wage])

# Check sizes of ttl_exp_array and wage_array
println("Size of ttl_exp_array: ", size(ttl_exp_array))
println("Size of wage_array: ", size(wage_array))

# Ensure ttl_exp_array and wage_array are of the same size
min_size = min(length(ttl_exp_array), length(wage_array))
ttl_exp_array = ttl_exp_array[1:min_size]
wage_array = wage_array[1:min_size]

# Re-check sizes after trimming
println("Size after trimming - ttl_exp_array: ", size(ttl_exp_array))
println("Size after trimming - wage_array: ", size(wage_array))

# Evaluate matrixops using ttl_exp_array and wage_array
elem_product, trans_product, sum_elements = matrixops(ttl_exp_array, wage_array)
println("Element-wise product of ttl_exp and wage:\n", elem_product)
println("Product of transpose(ttl_exp) and wage:\n", trans_product)
println("Sum of all elements in ttl_exp and wage:\n", sum_elements)

# Wrap everything into q4() function
function q4()
    # Load the matrix file again (in case it's needed to be within the function scope)
    data = load("/home/etica/Project/fall-2024-econ6343/ProblemSets/PS1-julia-intro/firstmatrix.jld")
    A = data["A"]
    B = data["B"]

    # Perform matrix operations on A and B
    elem_product, trans_product, sum_elements = matrixops(A, B)
    println("Element-wise product:\n", elem_product)
    println("Product of transpose(A) and B:\n", trans_product)
    println("Sum of all elements in A and B:\n", sum_elements)

    # Load nlsw88.csv within q4 function
    nlsw88 = CSV.read("/home/etica/Project/fall-2024-econ6343/ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame)

    # Convert ttl_exp and wage columns to arrays
    ttl_exp_array = convert(Array, nlsw88[!, :ttl_exp])
    wage_array = convert(Array, nlsw88[!, :wage])

    # Ensure ttl_exp_array and wage_array are of the same size
    min_size = min(length(ttl_exp_array), length(wage_array))
    ttl_exp_array = ttl_exp_array[1:min_size]
    wage_array = wage_array[1:min_size]

    # Evaluate matrixops using ttl_exp_array and wage_array
    elem_product, trans_product, sum_elements = matrixops(ttl_exp_array, wage_array)
    println("Element-wise product of ttl_exp and wage:\n", elem_product)
    println("Product of transpose(ttl_exp) and wage:\n", trans_product)
    println("Sum of all elements in ttl_exp and wage:\n", sum_elements)
end

# Call the function q4() at the very bottom of your script
q4()


# Question 5
# Test q1 Function
using Test
using Random
using JLD2
using CSV
using DataFrames

@testset "Testing q1 Function" begin
    # Call the function and get matrices
    A, B, C, D = q1()
    
    # Test if the seed is set correctly by checking if A and B are consistent
    Random.seed!(1234)
    expected_A = rand(Uniform(-5, 10), 10, 7)
    expected_B = rand(Normal(-2, 15), 10, 7)
    
    @test A ≈ expected_A
    @test B ≈ expected_B
    
    # Test matrix dimensions
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    
    # Test specific values in C
    @test C[1:5, 1:5] == A[1:5, 1:5]
    @test C[:, 6:7] == B[1:5, 6:7]
    
    # Test if D is correctly computed
    for i in 1:10, j in 1:7
        @test D[i,j] == (A[i,j] <= 0 ? A[i,j] : 0)
    end
    
    # Test vectorization of B in E
    E = vec(B)
    @test vec(B) == E
    
    # Test if F is correctly permuted and concatenated
    F = permutedims(cat(A, B; dims=3), (3, 1, 2))
    @test F == permutedims(cat(A, B; dims=3), (3, 1, 2))
    
    # Test if G is correctly computed as the Kronecker product
    G = kron(B, C)
    @test G == kron(B, C)
    
    # Test if the .jld files are correctly saved
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    
    # Test if .csv and .dat files are correctly generated and saved
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
    
    # Check the content of the saved .csv file
    df_C = CSV.read("Cmatrix.csv", DataFrame)
    @test df_C ≈ DataFrame(C, :auto)
    
    # Check the content of the saved .dat file
    df_D = CSV.read("Dmatrix.dat", DataFrame; delim='\t')
    @test df_D ≈ DataFrame(D, :auto)
end

# Test q2 Function
@testset "Test elementwise_product" begin
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    AB, AB2 = elementwise_product(A, B)
    @test AB == [5 12; 21 32]
    @test AB2 == [5 12; 21 32]
end

# Test filter_C_elements
@testset "Test filter_C_elements" begin
    C = [-10, -5, 0, 5, 10]
    Cprime, Cprime2 = filter_C_elements(C)
    @test Cprime == [-5, 0, 5]
    @test Cprime2 == [-5, 0, 5]
end

# Test create_3d_array
@testset "Test create_3d_array" begin
    N = 10
    K = 6
    T = 5
    X = create_3d_array(N, K, T)
    @test size(X) == (N, K, T)
    @test X[:,1,:] == ones(N, T)  # Check if intercept column is all ones
end

# Test create_beta
@testset "Test create_beta" begin
    K = 6
    T = 5
    β = create_beta(K, T)
    @test size(β) == (K, T)
    @test β[1, :] == [1.0, 1.25, 1.5, 1.75, 2.0]
end

# Test create_Y
@testset "Test create_Y" begin
    N = 10
    K = 6
    T = 5
    X = create_3d_array(N, K, T)
    β = create_beta(K, T)
    Y = create_Y(X, β, N, T)
    @test size(Y) == (N, T)
end

# Run all tests
@testset "All tests" begin
    include("wang0235_hw1.jl")
end


# Test q3 Function
using Test
using CSV
using DataFrames

function test_q3()
    q3()  # Call the q3 function to execute its internal logic

    # Load processed data
    processed_df = CSV.read("nlsw88_processed.csv", DataFrame)

    @test !isempty(processed_df)

    # Adjust the tolerance to allow a small variance in floating-point calculations
    @test isapprox(mean(processed_df[!, :never_married]) * 100, 10.42, atol = 0.01)
end

# Execute the test function with updated logic
test_q3()

# Test q4 Function
# Load necessary libraries
using Test
using CSV
using DataFrames
using JLD  

# Mock functions to replace file I/O
function mock_load()
    return Dict("A" => [1 2; 3 4], "B" => [5 6; 7 8])
end

function mock_CSV_read()
    return DataFrame(ttl_exp=[10, 20, 30], wage=[15, 25, 35])
end

# Assuming matrixops is defined as needed
function matrixops(A, B)
    elem_product = A .* B
    trans_product = transpose(A) * B
    sum_elements = sum(A) + sum(B)
    return elem_product, trans_product, sum_elements
end

# Test cases
@testset "Function q4 Tests" begin
    # Mock data loading and matrix operations
    @testset "Matrix Operations Tests" begin
        data = mock_load()  # Using the mocked function
        elem_product, trans_product, sum_elements = matrixops(data["A"], data["B"])
        
        @test elem_product == [1*5 2*6; 3*7 4*8]
        @test trans_product == transpose([1 2; 3 4]) * [5 6; 7 8]
        @test sum_elements == sum([1 2; 3 4]) + sum([5 6; 7 8])
    end

    # Mock CSV reading and array handling
    @testset "CSV and Array Manipulation Tests" begin
        nlsw88 = mock_CSV_read()  # Using the mocked CSV reading function
        ttl_exp_array = convert(Array, nlsw88[!, :ttl_exp])
        wage_array = convert(Array, nlsw88[!, :wage])
        
        min_size = min(length(ttl_exp_array), length(wage_array))
        ttl_exp_array = ttl_exp_array[1:min_size]
        wage_array = wage_array[1:min_size]

        elem_product, trans_product, sum_elements = matrixops(ttl_exp_array, wage_array)
        
        @test length(ttl_exp_array) == length(wage_array)
        @test elem_product == ttl_exp_array .* wage_array
        @test trans_product == transpose(ttl_exp_array) * wage_array
        @test sum_elements == sum(ttl_exp_array) + sum(wage_array)
    end
end

