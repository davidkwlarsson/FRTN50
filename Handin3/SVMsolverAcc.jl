using ProximalOperators, LinearAlgebra, Plots, Random, Statistics

include("../Handin2/problem.jl")

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


function prox_grad_method(x,y, beta_method = 0, mu = 0)
    ITER = 100000
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    # g = SqrNormL2(1/(2*lambda))
    dual = randn(N)
    dual_old = dual
    res = zeros(ITER)
    beta = 0
    if beta_method == 2
        beta = (1 - sqrt(mu*gamma)) / (1 + sqrt(mu*gamma))
    end
    for i = 0:ITER-1
        if beta_method == 1
            beta = (i - 2) / (i + 1)
        end
        y_acc = dual + beta*(dual - dual_old)
        grad_g = Q * dual
        y_k = dual - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        res[i+1] = norm(dual .- dual_1, 2)
        if i % 1000 == 0
            #println(res)
        end
        dual_old = dual
        dual = dual_1
    end

    return dual
end

function testSVM()
    dual = prox_grad_method(x_train, y_train)
    y_pred = similar(y_test)
    for i = 1:length(x_test)
        y_pred[i] = prediction(dual, x_test[i,:])
    end
    naive_class = 1
    naive_errors = sum(abs.(y_test.-naive_class))/2
    errors = sum(abs.(y_test.-y_pred))/2
    #y_pred_train = similar(y_train)
    #for i = 1:length(x_train)
    #    y_pred_train[i] = prediction(dual, x_train[i,:])
    #end
    println(errors, " errors out of ", length(x_test),
        ", naive classifier errors : " , naive_errors)
    #train_errors = sum(abs.(y_train .- y_pred_train))/2
    #println("Train error = ", train_errors, " out of : ", length(y_train))
    return errors
end

lambda = 0.0001
sigma = 0.5

#configuration of (lambda, sigma) = (0.1, 2) (0.001, 0.5) (0.00001, 0.25)
println("lambda : " ,lambda,"     sigma: ", sigma)
x_test, y_test = svm_test_1() #best = (0.0001, 0.5)
testSVM()
