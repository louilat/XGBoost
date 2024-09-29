using Printf
using Plots
include("data/generate_data.jl")
include("tree/larger_tree.jl")

println("Generating data ...")
X, y = generate_complex_data(Int32(10000))
println("Done!")

println("Initializing tree ...")
tree::DecisionTree = DecisionTree(50, 3, 2, nothing)
println("Done!")

println("Training tree ...")
fit(tree, X = X, y = y)
println("Done!")

println("Predict ...")
X_test, y_test = generate_complex_data(Int32(1000))
y_pred = predict(tree, X_test)
println("Done!")

plot(X_test[:, 1] - X_test[:, 2], y_test, seriestype = :scatter)
plot!(X_test[:, 1] - X_test[:, 2], y_pred, seriestype = :scatter)
