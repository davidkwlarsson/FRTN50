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

function prox_grad_CGD(x, y, lambda, sigma, ITER, dual, gamma_method = 1)
    K = GaussKernelMatrix(x, sigma)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    # Blir matten rätt här?
    h1 = Conjugate(HingeLoss(ones(1),1))
    # g = SqrNormL2(1/(2*lambda))
    res = zeros(Int(round(ITER/N + 1)))
    solutions_save = zeros(N, Int(round(ITER/N)) + 1)
    solutions_save[:,1] = dual
    idx = rand(1:N,ITER)
    if gamma_method == 2
        gamma = 1/norm(Q,2)
    end
    ii = 1
    for i = 1:ITER
        j = idx[i]
        if gamma_method == 1
             gamma = 1/Q[j,j]
        end
        grad_gj = transpose(Q[j,:]) * dual
        y_kj = [dual[j] - gamma*grad_gj]
        dual_1j, hw =  prox(h1, y_kj, gamma)

        if i % (N*100) == 0
            # res[ii] = norm(dual[j] - dual_1j[1], 1)
            solutions_save[:,ii] = dual
            ii += 1
        end
        # We could use only dual and dual_1 but that would maybe be more confusing
        dual[j] = dual_1j[1]
    end

    return dual, res, solutions_save[:,1:ii]
end

function prox_grad(x, y, lambda, sigma, ITER, dual)
    K = GaussKernelMatrix(x, sigma)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    res = zeros(ITER)
    solutions_save = zeros(length(x), ITER)
    for i = 1:ITER
        grad_g = Q * dual
        y_k = dual - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        res[i] = norm(dual .- dual_1, 2)
        solutions_save[:,i] = dual_1
        dual = dual_1
    end
    return dual, res, solutions_save
end

function testSVM(x, y)
    # initiate nessecary data and hyperparameters
    lambda = 10^-3
    sigma = 0.5
    ITER = 10^5
    N = length(x)
    initial_point = randn(N)
    beta_method = 0
    mu = 0
    acc = false
    cgd = true
    # step 1: solve the problem to a high precision and store the solution
    dual, stepsize, dual_list0 = prox_grad(x, y, lambda, sigma, ITER, initial_point)
    print("stepsize after 10^5 iterations: ", stepsize[end])

    solnorm0 = ones(ITER)
    for i = 1:ITER
        solnorm0[i] = norm(dual_list0[:,i] .- dual, 2)
    end
    display(plot(solnorm0,
    title="Comparation of different accelerators \n lambda = 10^-3, sigma = 0.5",
    xlabel= "# Iterations",
    ylabel="||x_optimal - x_k||",
    label="beta0",
    yaxis=(:log, (10^-20, 10))))

    if cgd
        ITER = 5*10^7
        gamma_method = 1
        _, _, dual_listCGD1 = prox_grad_CGD(x, y, lambda, sigma, ITER, initial_point, gamma_method)
        n = length(dual_listCGD1[1,:])
        solnormCGD1 = ones(n)
        ls = ones(n)
        for i = 1:n
            solnormCGD1[i] = norm(dual_listCGD1[:,i] .- dual, 2)
            ls[i] = 100*i
        end
        display(plot!(ls, solnormCGD1,
        label="gamma1",
        yaxis=(:log, (10^-20, 10))))

        gamma_method = 2
        _, _, dual_listCGD2 = prox_grad_CGD(x, y, lambda, sigma, ITER, initial_point, gamma_method)
        solnormCGD2 = ones(n)
        for i = 1:n
            solnormCGD2[i] = norm(dual_listCGD2[:,i] .- dual, 2)
        end
        display(plot!(ls, solnormCGD2,
        label="gamma2",
        yaxis=(:log, (10^-20, 10))))

    end

    # step 2: try the three beta_methods™ and plot Plot ||x^* - x^k||
    if acc
        beta_method = 1
        mu = 0
        _, _, dual_list1 = prox_grad_method(x, y, lambda, sigma, ITER, initial_point, beta_method, mu)
        solnorm1 = ones(ITER)
        for i = 1:ITER
            solnorm1[i] = norm(dual_list1[:,i] .- dual, 2)
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
            solnorm2[i] = norm(dual_list2[:,i] .- dual, 2)
        end
        display(plot!(solnorm2,
        label="beta2",
        yaxis=(:log, (10^-20, 1))))
    end
end

#always gives the same values bc mt
x, y = svm_train()
testSVM(x, y)

#savefig("task1.png")
