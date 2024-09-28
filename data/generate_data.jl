

function generate_data(n::Int32)::Tuple{Array{Float32}, Array{Float32}}
	X::Array{Float32} = randn(n, 10)
	y::Array{Float32} = 1 ./ (1 .+ exp.(.-X[:, 5])) .+ 0.1 * randn(n)
	return X, y
end
