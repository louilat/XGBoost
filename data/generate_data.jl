

function generate_easy_data(n::Int32)::Tuple{Array{Float32}, Array{Float32}}
	X::Array{Float32} = randn(n, 1)
	y::Array{Float32} = 1 ./ (1 .+ exp.(.-X[:, 1])) + 0.05 .* randn(n)
	return X, y
end

function generate_easy_data(X::Array{Float32})::Array{Float32}
	y::Array{Float32} = 1 ./ (1 .+ exp.(.-X[:, 1])) + 0.05 .* randn(size(X)[1])
	return y
end

function generate_complex_data(n::Int32)::Tuple{Array{Float32}, Array{Float32}}
	X::Array{Float32} = 5 .* randn(n, 2)
	y::Array{Float32} = 1 ./ (1 .+ exp.(.-X[:, 1] + X[:, 2])) .+ 0.1 * randn(n)
	return X, y
end
