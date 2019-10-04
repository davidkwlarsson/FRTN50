using ProximalOperators, LinearAlgebra, Plots, Random, Statistics

include("../Handin2/problem.jl")

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

# Never used
function GaussKernel(x,y, sigma)
    return exp(-1/(2*sigma^2)*norm(x .- y,2)^2)
end


function prox_grad_method(x, y, lambda, sigma, ITER, dual, beta_method = 0, mu = 0)
    K = GaussKernelMatrix(x, sigma)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    # g = SqrNormL2(1/(2*lambda))
    dual_old = dual
    res = zeros(ITER)
    solutions_save = zeros(length(x), ITER)
    beta = 0
    if beta_method == 2
        beta = (1 - sqrt(mu*gamma)) / (1 + sqrt(mu*gamma))
    end
    for i = 0:ITER-1
        if beta_method == 1
            beta = (i - 2) / (i + 1)
        end
        dual_acc = dual + beta*(dual - dual_old)
        grad_g = Q * dual_acc
        y_k = dual_acc - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        # Not the resudual we ultimatly want for this task
        res[i+1] = norm(dual .- dual_1, 2)
        solutions_save[:,i+1] = dual_1
        if i % 1000 == 0
            #println(res)
        end
        # We could use only dual and dual_1 but that would maybe be more confusing
        dual_old = dual
        dual = dual_1
    end

    return dual, res, solutions_save
end

function testSVM(x, y)
    # initiate nessecary data and hyperparameters
    lambda = 10^-3
    sigma = 0.5
    ITER = 10^5
    initial_point = randn(length(x))
    beta_method = 0
    mu = 0
    # step 1: solve the problem to a high precision and store the solution
    dual, stepsize, dual_list0 = prox_grad_method(x, y, lambda, sigma, ITER, initial_point, beta_method, mu)
# prox_grad_method(x, y, lambda, sigma, ITER, dual, beta_method = 0, mu = 0)
    # plot(stepsize[1000:end], yaxis=log10)
    print("stepsize after 10^5 iterations: ", stepsize[end])
    # stepsize[end] = 2.0342000156412462e-18, we can conclude convergence.

    # step 2: try the three beta_methods™ and plot Plot ||x^* - x^k||
    #ITER = 10^5
    true_dual = dual * ones(1,ITER)
    solnorm0 = ones(ITER)
    for i = 1:ITER
        solnorm0[i] = norm(dual_list0[:,i] .- true_dual[:,i], 2)
    end
    display(plot(solnorm0,
    title="Comparation of different accelerators \n lambda = 10^-3, sigma = 0.5",
    xlabel= "# Iterations",
    ylabel="||x_optimal - x_k||",
    label="beta0",
    yaxis=(:log, (10^-20, 10))))

    beta_method = 1
    mu = 0
    _, _, dual_list1 = prox_grad_method(x, y, lambda, sigma, ITER, initial_point, beta_method, mu)
    solnorm1 = ones(ITER)
    for i = 1:ITER
        solnorm1[i] = norm(dual_list1[:,i] .- true_dual[:,i], 2)
    end
    display(plot!(solnorm1,
    label="beta1",
    yaxis=(:log, (10^-20, 10))))

    beta_method = 2
    mu = 30
    # mu = 30: 1800 iter till e-15
    _, _, dual_list2 = prox_grad_method(x, y, lambda, sigma, ITER, initial_point, beta_method, mu)
    solnorm2 = ones(ITER)
    for i = 1:ITER
        solnorm2[i] = norm(dual_list2[:,i] .- true_dual[:,i], 2)
    end
    display(plot!(solnorm2,
    label="beta2",
    yaxis=(:log, (10^-20, 1))))
end

#always gives the same values bc mt
x, y = svm_train()
testSVM(x, y)
savefig("task1.png")
