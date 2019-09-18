using LinearAlgebra, Plots

include("functions.jl")
include("problem.jl")

Q,q,a,b = problem_data()

function dual_gradient(Q,q,a,b,gamma)
    ITER = 100
    x_k = randn(length(a))
    res = zeros(ITER)
    L = norm(Q,2)
    x_k1 = similar(x_k)

    for i = 1:ITER
        x_k1 = prox_boxconj(x_k - gamma*grad_quadconj(x_k,Q,q), a, b, gamma)
        res[i] = norm(x_k1 - x_k,2)
        x_k = x_k1
    end
end

function proxi_gradient(Q,q,a,b, gamma)
    ITER = 100
    x_k = randn(length(a))
    res = zeros(ITER)
    x_k1 = similar(x_k)
    f_x = zeros(ITER)

    for i = 1:ITER
        y_k = x_k - gamma*grad_quad(x_k,Q,q)
        x_k1 = prox_box(y_k, a, b, gamma)
        res[i] = norm(x_k1 - x_k,2)
        x_k = x_k1
        f_x[i] = quad(x_k, Q, q)
    end

    return x_k,res,f_x
end

function plot_res(gamma = LinRange(0,2/L,10))
    L = norm(Q,2)
    n_g = length(gamma)
    res = zeros(100,n_g)
    x_star = zeros(length(a),n_g)
    f_x = zeros(100,n_g)
    for i = 1:n_g
        x_star[:,i],res[:,i] = proxi_gradient(Q,q,a,b,gamma[i])
    end
    plot(res)
end

function plot_conj(gamma = LinRange(0,2/L,10))
    L = norm(inv(Q),2)
    n_g = length(gamma)
    res_conj = zeros(100,n_g)
    f_x = zeros(100, n_g)
    x_star = zeros(length(a),n_g)
    for i = 1:n_g
        my_star,res_conj[:,i] = proxi_gradient(Q,q,a,b,gamma[i])
        x_star[:,i] = dual2primal(my_star,Q,q,a,b)
    end
    plot(res_conj)
end
