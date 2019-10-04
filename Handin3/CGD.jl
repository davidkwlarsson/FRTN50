
using ProximalOperators, LinearAlgebra, Plots, Random, Statistics


include("problem.jl")
x_train, y_train = svm_train()



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


function coord_prox_grad_method(x,y)
    ITER = 100000
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(1),1/N))
    dual = randn(N)
    dual_prev = zeros(N)
    res = zeros(0)
    delta = zeros(N)
    tk = 0
    for i = 1:ITER
        index = rand(big.(1:N))
        dual_i = dual[index]
        grad_g = Q * dual
        y_k = dual_i - gamma*grad_g[index]
        dual_1, hw =  prox(h1, [y_k], gamma)
        if i % 1000 == 0
            delta = dual .- dual_prev
            print(norm(delta,2))
            append!(res, norm(delta,2))
            dual_prev = dual

        end
        dual[index] = dual_1[1]
    end
    print(res)
    display(plot(res, yaxis =(:log, (10^(-15), 1))))
    return dual, res
end

lambda = 0.0001
sigma = 0.5

coord_prox_grad_method(x_train, y_train)
