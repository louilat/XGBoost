using Printf
include("data/generate_data.jl")
include("tree/tree.jl")

println("Generating data ...")
X, y = generate_data(Int32(10000))
println("Done!")

println("Initializing tree ...")
tree = TreeNode()
println("Done!")

println("Training tree ...")
@time train_tree(tree, X_train = X, y_train = y)
println("Done!")

println("Training tree 2 ...")
@time train_tree_2(tree, X_train = X, y_train = y)
println("Done!")

