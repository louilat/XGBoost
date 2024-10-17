using Printf
using Statistics
include("tree_regressor.jl")

mutable struct XgbTree
	min_samples_split::Int16
	max_depth::Int16
	n_features::Int32
	lambda::Float32
	root::Any
end

function XgbTree(min_samples_split::Int16, max_depth::Int16, n_features::Int32, lambda::Float32)::XgbTree
	return XgbTree(min_samples_split, max_depth, n_features, lambda, nothing)
end


mutable struct XgbRegressor
	lambda::Float32
	min_samples_split::Int16
	max_depth::Int16
	n_features::Int32
	n_estimators::Int16
	learning_rate::Float32
	model::Array{XgbTree}
	mean::Float32
end

function fit(xgb_model::XgbRegressor; X::Array{Float32}, y::Array{Float32})
	model::Array{XgbTree} = Array{XgbTree}(undef, xgb_model.n_estimators)
	y_pred::Array{Float32} = fill(mean(y), size(y)[1])
	residuals::Array{Float32} = y - y_pred
	xgb_model.mean = mean(y)
	for k in 1:xgb_model.n_estimators
		current_tree::XgbTree = XgbTree(xgb_model.min_samples_split, xgb_model.max_depth, xgb_model.n_features, xgb_model.lambda)
		fit(current_tree, X = X, y = residuals)
		y_pred = y_pred + xgb_model.learning_rate * predict(current_tree, X)
		residuals = y - y_pred
		model[k] = current_tree
	end
	xgb_model.model = model
end

function fit(xgb_tree::XgbTree; X::Array{Float32}, y::Array{Float32})
	@assert xgb_tree.n_features <= size(X)[2]
	xgb_tree.root = grow_tree(xgb_tree, X, y, Int16(0))
end

function grow_tree(xgb_tree::XgbTree, X::Array{Float32}, y::Array{Float32}, depth::Int16)
	n_samples::Int32 = size(X)[1]

	# Stopping criteria
	if depth > xgb_tree.max_depth || n_samples < xgb_tree.min_samples_split
		leaf_value::Float32 = mean(y)
		return Node(leaf_value)
	end

	# Find the best split
	features_idx::Array{Int16} = sample(1:size(X)[2], xgb_tree.n_features, replace = false)
	best_feature::Int16, best_threshold::Float32 = find_best_split_xgb(X, y, features_idx, xgb_tree.lambda)
	feature = @view X[:, best_feature]

	# Create child nodes
	idx_left, idx_right = findall(<=(best_threshold), feature), findall(>(best_threshold), feature)
	X_left::Array{Float32}, X_right::Array{Float32} = X[idx_left, :], X[idx_right, :]
	y_left::Array{Float32}, y_right::Array{Float32} = y[idx_left], y[idx_right]
	left::Node = grow_tree(xgb_tree, X_left, y_left, Int16(depth + 1))
	right::Node = grow_tree(xgb_tree, X_right, y_right, Int16(depth + 1))
	return Node(best_feature, best_threshold, left, right)
end

function find_best_split_xgb(X::Array{Float32}, y::Array{Float32}, features_idx::Array{Int16}, lambda::Float32)::Tuple{Int16, Float32}
	n = size(X)[1]
	p = length(features_idx)
	ordered_indexes::Array{Int32} = Array{Int32}(undef, n)
	best_similarity_score::Float32 = var(y)
	similarity_score::Float32 = var(y)
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
			similarity_score = delta_similarity_score(y, y_group1, y_group2, lambda)
			if similarity_score > best_similarity_score
				best_similarity_score = similarity_score
				best_feature = k
				best_value = i
			end
		end
		println("      --> best_similarity_score = $best_similarity_score")
	end
	return best_feature, X[best_value, best_feature]
end


function delta_similarity_score(y::Array{Float32}, y_left::SubArray{Float32}, y_right::SubArray{Float32}, lambda::Float32)::Float32
	sim_score::Float32 = sum(y)^2 / (sum(y) + lambda)
	sim_score_left::Float32 = sum(y_left)^2 / (length(y_left) + lambda)
	sim_score_right::Float32 = sum(y_right)^2 / (length(y_right) + lambda)
	return sim_score_left + sim_score_right - sim_score
end

function predict(xgb_tree::XgbTree, X::Array{Float32})::Array{Float32}
	n::Int32 = size(X)[1]
	y_pred::Array{Float32} = Array{Float32}(undef, n)
	for i in 1:n
		x = @view X[i, :]
		y_pred[i] = traverse_tree(xgb_tree, x)
	end
	return y_pred
end

function traverse_tree(xgb_tree::XgbTree, x::SubArray{Float32})::Float32
	current_node = xgb_tree.root
	while !is_leaf_node(current_node)
		if x[current_node.feature] <= current_node.threshold
			current_node = current_node.left
		else
			current_node = current_node.right
		end
	end
	return current_node.value
end

function predict(xgb_model::XgbRegressor, X::Array{Float32})::Array{Float32}
	y_pred::Array{Float32} = fill(xgb_model.mean, size(X)[1])
	for k in 1:xgb_model.n_estimators
		y_pred = y_pred + xgb_model.learning_rate * predict(xgb_model.model[k], X)
	end
	return y_pred
end
