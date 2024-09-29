using Printf
using Plots
include("data/generate_data.jl")
include("tree/tree_regressor.jl")

println("Generating data ...")
X, y = generate_easy_data(Int32(10000))
println("Done!")

println("Initializing tree ...")
tree::DecisionTree = DecisionTree(50, 3, 1, nothing)
println("Done!")

println("Training tree ...")
fit(tree, X = X, y = y)
println("Done!")

println("Predict ...")
# X_test, y_test = generate_easy_data(Int32(1000))
X_test::Array{Float32} = reshape(LinRange(-3, 3, 1000), 1000, 1)
y_test = generate_easy_data(X_test)
y_pred = predict(tree, X_test)
println("Done!")

scatter(X_test[:, 1], y_test, label = "train set")
plot!(X_test[:, 1], y_pred, lw = 3, label = "model prediction")
