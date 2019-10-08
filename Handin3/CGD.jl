
using ProximalOperators, LinearAlgebra, Plots, Random, Statistics


include("problem.jl")
include("SVMsolver.jl")

function GaussKernelMatrix(x)
    N = length(x)
    K = zeros(N,N)
    for i = 1:N
        for j = 1:N
            K[i,j] = exp(-1/(2*sigma^2)*norm(x[i,:] - x[j,:], 2)^2)
        end
    end
    return K
end

function GaussKernel(x,y)
    return exp(-1/(2*sigma^2)*norm(x .- y,2)^2)
end


function coord_prox_grad_method(dual_init,x,y,C_grad = false,dual_sol = 0, have_sol = false, ITER = 1000)
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(1),1/N))
    dual = copy(dual_init)
    dual_prev = copy(dual)
    res = zeros(0)
    x_plot = zeros(0)
    delta = zeros(N)
    tk = 0
    for i = 1:ITER
        index = rand(big.(1:N))
        dual_i = dual[index]
        if C_grad
            gamma = 1/(Q[index,index])
        end
        grad_g = (Q[index,:])' * dual
        y_k = [dual_i - gamma*grad_g]
        dual_1, hw =  prox(h1, y_k, gamma)
        dual[index] = dual_1[1]
        if i % 100 == 0
            if have_sol
                append!(res, norm(dual .- dual_sol,2))
            else
                append!(res, norm(dual .- dual_prev,2))
            end
            append!(x_plot, i/N)
            dual_prev = copy(dual)
        end
    end
    display(plot(res, yaxis=("|| x^k - x* ||",:log),
            xlabel = "Iterations"))
    return dual, res
end

x_train, y_train = svm_train()
lambda = 0.001
sigma = 0.5
dual_init = randn(length(y_train))

dual_sol, res= prox_grad_method(dual_init, x_train, y_train, 0, 0, 0, false, 100000)

coord_grad = false
coord_prox_grad_method(dual_init,x_train, y_train, coord_grad, dual_sol, true,1000000)
