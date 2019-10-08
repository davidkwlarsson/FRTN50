using ProximalOperators, LinearAlgebra, Plots, Random, Statistics


include("problem.jl")



function GaussKernelMatrix(x, sigma)
    N = length(x)
    K = zeros(N,N)
    for i = 1:N
        for j = 1:N
            K[i,j] = exp(-1/(2*sigma^2)*norm(x[i,:] - x[j,:], 2)^2)
        end
    end
    return K
end

function GaussKernel(x,y,sigma)
    return exp(-1/(2*sigma^2)*norm(x .- y,2)^2)
end


function prox_grad_method(dual_init,x,y,beta_method = 1, mu = 0, dual_sol = 0, have_sol = false, ITER = 10000)
    K = GaussKernelMatrix(x,sigma)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    res = zeros(ITER)
    if beta_method == 3
        beta_k = (1 - sqrt(mu*gamma))/(1 + sqrt(mu*gamma))
        print("this is beta:",  beta_k, "              this is gamma: ", gamma)
    end
    delta = zeros(N)
    tk = 0
    dual = copy(dual_init)
    for i = 1:ITER
        if beta_method == 1
            beta_k = (i-3)/i
            # print(beta_k)
        elseif beta_method == 2
            tk1 = (1+sqrt(1+4*(tk^2)))/2
            beta_k = (tk - 1)/tk1
            tk = tk1
            # print(beta_k)
        end
        if beta_method != 0
            # print("Beta_k = : " , beta_k, "\n")
            dual_12 = dual + beta_k*delta
        else
            dual_12 = dual
        end
        grad_g = Q * dual_12
        y_k = dual_12 - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        delta = dual_1 - dual
        if have_sol
            res[i] = norm(dual_1 .- dual_sol, 2)
        else
            res[i] = norm(delta, 2)
        end
        dual = dual_1
    end
    if have_sol
        display(plot(res, yaxis=("|| x^k - x* ||",:log),
                xlabel = "Iterations", title = "Beta method : " * string(beta_method)))
    else
        display(plot(res, yaxis=("|| x^k+1 - x^k ||",:log),
                xlabel = "Iterations", title = "Beta method : " * string(beta_method)))
    end
    return dual, res
end

x_train, y_train = svm_train()
lambda = 0.001
sigma = 0.5


# NOTES : configuration of (lambda, sigma) = (0.1, 2) (0.001, 0.5) (0.00001, 0.25)
# println("lambda : " ,lambda,"     sigma: ", sigma)
dual_init = randn(length(x_train))
dual_sol, res= prox_grad_method(dual_init, x_train, y_train, 0, 0, 0, false, 100000)



prox_grad_method(dual_init, x_train, y_train, 1, 30, dual_sol, true, 10000)
