using Printf
using Statistics
using StatsBase

mutable struct Node
	feature::Any
	threshold::Any
	left::Any
	right::Any
	value::Any
end

function Node(value::Float32)::Node
	return Node(nothing, nothing, nothing, nothing, value)
end

function Node(feature::Int16, threshold::Float32, left::Node, right::Node)::Node
	return Node(feature, threshold, left, right, nothing)
end

function is_leaf_node(node::Node)::Bool
	return !isnothing(node.value)
end

mutable struct DecisionTree
	min_samples_split::Int16
	max_depth::Int16
	n_features::Int32
	root::Any
end

function DecisionTree()
	return DecisionTree(10, 5, 6, nothing)
end

function fit(decision_tree::DecisionTree; X::Array{Float32}, y::Array{Float32})
	@assert decision_tree.n_features <= size(X)[2]
	decision_tree.root = grow_tree(decision_tree, X, y, Int16(0))
end

function grow_tree(decision_tree::DecisionTree, X::Array{Float32}, y::Array{Float32}, depth::Int16)
	n_samples::Int32 = size(X)[1]

	# Stopping criteria
	if depth > decision_tree.max_depth || n_samples < decision_tree.min_samples_split
		leaf_value::Float32 = mean(y)
		return Node(leaf_value)
	end

	# Find the best split
	features_idx::Array{Int16} = sample(1:size(X)[2], decision_tree.n_features, replace = false)
	best_feature::Int16, best_threshold::Float32 = find_best_split(X, y, features_idx)
	feature = @view X[:, best_feature]

	# Create child nodes
	idx_left, idx_right = findall(<=(best_threshold), feature), findall(>(best_threshold), feature)
	X_left::Array{Float32}, X_right::Array{Float32} = X[idx_left, :], X[idx_right, :]
	y_left::Array{Float32}, y_right::Array{Float32} = y[idx_left], y[idx_right]
	left::Node = grow_tree(decision_tree, X_left, y_left, Int16(depth + 1))
	right::Node = grow_tree(decision_tree, X_right, y_right, Int16(depth + 1))
	return Node(best_feature, best_threshold, left, right)
end

function find_best_split(X::Array{Float32}, y::Array{Float32}, features_idx::Array{Int16})::Tuple{Int16, Float32}
	n = size(X)[1]
	p = length(features_idx)
	ordered_indexes::Array{Int32} = Array{Int32}(undef, n)
	max_var_red::Float32 = typemin(Float32)
	cur_var_red::Float32 = typemin(Float32)
	best_feature::Int16 = 1
	best_value::Int32 = 1
	for k in features_idx
		println("   Analysing feature $k / $p ...")
		feature = @view X[:, k]
		ordered_indexes = sortperm(feature)
		ordered_y = @view y[ordered_indexes]
		for i in 2:n-1
			y_group1 = @view ordered_y[1:i]
			y_group2 = @view ordered_y[i+1:n]
			cur_var_red = variance_reduction(y, y_group1, y_group2)
			if cur_var_red > max_var_red
				max_var_red = cur_var_red
				best_feature = k
				best_value = i
			end
		end
		println("      --> max_var_red = $max_var_red")
	end
	return best_feature, X[best_value, best_feature]
end

function variance_reduction(y::Array{Float32}, y_left::SubArray{Float32}, y_right::SubArray{Float32})::Float32
	weight_l = length(y_left) / length(y)
	weight_r = length(y_right) / length(y)
	return var(y) - weight_l * var(y_left) - weight_r * var(y_right)
end

function predict(decision_tree::DecisionTree, X::Array{Float32})::Array{Float32}
	n::Int32 = size(X)[1]
	y_pred::Array{Float32} = Array{Float32}(undef, n)
	for i in 1:n
		x = @view X[i, :]
		y_pred[i] = traverse_tree(decision_tree, x)
	end
	return y_pred
end

function traverse_tree(decision_tree::DecisionTree, x::SubArray{Float32})::Float32
	current_node = decision_tree.root
	while !is_leaf_node(current_node)
		if x[current_node.feature] <= current_node.threshold
			current_node = current_node.left
		else
			current_node = current_node.right
		end
	end
	return current_node.value
end
