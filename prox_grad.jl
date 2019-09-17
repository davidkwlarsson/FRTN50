using LinearAlgebra

include("functions.jl")
include("problem.jl")

Q,q,a,b = problem_data()

function dual_gradient(Q,q,a,b)
    ITER = 100
    mt = MersenneTwister(123)
    x_k = randn(mt,length(a))
    res = zeros(ITER)
    L = norm(Q,2)
    gamma = range(0,stop = L/2, length = 10)
    x_k1 = similar(x_k)

    for i = 1:ITER
        x_k1 = prox_boxconj(x_k - gamma[4]*grad_quadconj(x_k,Q,q), a, b, gamma[4])
        res[i] = norm(x_k1 - x_k)
        x_k = x_k1
    end
end

function proxi_gradient(Q,q,a,b)
    ITER = 100
    x_k = randn(length(a))
    res = zeros(ITER)
    L = norm(Q,2)
    gamma = range(0,stop = L/2, length = 10)
    x_k1 = similar(x_k)

    for i = 1:ITER
        y_k = x_k - 1*grad_quad(x_k,Q,q)
        x_k1 = prox_box(y_k, a, b, 1)
        res[i] = norm(x_k1 - x_k,2)
        x_k = x_k1
    end

    return x_k,res
end

x_star,res = proxi_gradient(Q,q,a,b)
