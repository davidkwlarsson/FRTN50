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


function prox_grad_method(x,y,beta_method = 1, mu = 0)
    ITER = 100000
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    dual = randn(N)
    res = zeros(ITER)
    if beta_method == 3
        beta_k = (1-sqrt(mu*gamma))/(1+sqrt(mu*gamma))
    end
    delta = zeros(N)
    tk = 0
    for i = 1:ITER
        if beta_method == 1
            beta_k = (i-3)/(i+1)
        elseif beta_method == 2
            tk1 = (1+sqrt(1+4*(tk^2)))/2
            beta = (tk - 1)/tk1
        end
        dual = dual .+ beta_k.*delta
        grad_g = Q * dual
        y_k = dual - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        delta = dual .- dual_1
        res[i] = norm(delta, 2)
        dual = dual_1
    end
    display(plot(res, yaxis=:log))
    return dual, res
end


lambda = 0.0001
sigma = 0.5


#configuration of (lambda, sigma) = (0.1, 2) (0.001, 0.5) (0.00001, 0.25)
println("lambda : " ,lambda,"     sigma: ", sigma)
x_test, y_test = svm_test_1() #best = (0.0001, 0.5)
prox_grad_method(x_train, y_train, 3, 100000000)
