using LinearAlgebra

include("functions.jl")
include("problem.jl")

Q,q,a,b = problem_data()
ITER = 100
x_k = zeros((length(a)))
L = norm(inv(Q),2)
gamma = range(0,stop = L/2, length = 10)

for i = 1:ITER
    x_k1 = prox_boxconj(x_k - gamma)
end
