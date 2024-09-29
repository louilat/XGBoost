using Printf
using Statistics

mutable struct TreeNode
	feature::Int32
	value::Float32
	left::Float32
	right::Float32
end

function TreeNode()
	return TreeNode(1, 0, 0, 0)
end

function variance_reduction(y::Array{Float32}, y_left::SubArray{Float32}, y_right::SubArray{Float32})::Float32
	weight_l = length(y_left) / length(y)
	weight_r = length(y_right) / length(y)
	return var(y) - weight_l * var(y_left) - weight_r * var(y_right)
end

function train_tree(tree::TreeNode; X_train::Array{Float32}, y_train::Array{Float32})
	n, p = size(X_train)
	ordered_indexes::Array{Int32} = Array{Int32}(undef, n)
	max_var_red::Float32 = typemin(Float32)
	cur_var_red::Float32 = typemin(Float32)
	best_feature::Int32 = 1
	best_value::Int32 = 1
	p1::Float32 = 1
	p2::Float32 = 0
	for k in 1:p
		println("   Analysing feature $k / $p ...")
		feature = @view X_train[:, k]
		ordered_indexes = sortperm(feature)
		ordered_y = @view y_train[ordered_indexes]
		for i in 2:n-1
			y_group1 = @view ordered_y[1:i]
			y_group2 = @view ordered_y[i+1:n]
			p1 = mean(y_group1)
			p2 = mean(y_group2)
			cur_var_red = variance_reduction(y_train, y_group1, y_group2)
			if cur_var_red > max_var_red
				max_var_red = cur_var_red
				best_feature = k
				best_value = i
			end
		end
		println("      --> max_var_red = $max_var_red")
	end
	tree.feature = best_feature
	final_value::Float32 = X_train[best_value, best_feature]
	tree.value = final_value
	tree.left = mean(y_train[findall(<=(final_value), X_train[:, best_feature])])
	tree.right = mean(y_train[findall(>(final_value), X_train[:, best_feature])])
	println("Trained node: feature = $best_feature, value = $final_value")
end
