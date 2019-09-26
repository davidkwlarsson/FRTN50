using ProximalOperators, LinearAlgebra, Plots


include("problem.jl")

x_train, y_train = svm_train()
x_test, y_test = svm_test_1()
lambda = 0.01
sigma = 0.5

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


function prox_grad_method(x,y)
    lambda = 0.01
    ITER = 1
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),N))
    g = SqrNormL2(1/(2*lambda))
    dual = randn(N)
    res = zeros(ITER)
    for i = 1:ITER
        grad_g = 1/lambda * Q * dual
        y_k = dual - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        dual = dual_1
    end
    return dual
end

function prediction(dual, x)
    K = GaussKernel(x_train[1,:],x)
    for k = 2:length(x_train)
        K = cat(K, GaussKernel(x_train[k,:],x), dims = 1)
    end
    yK = y_train.*K
    pred = -1/lambda.*transpose(dual)*yK
    return sign(pred[1])
end


function testSVM()
    dual = prox_grad_method(x_train, y_train)
    y_pred = similar(y_test)
    for i = 1:length(x_test)
        y_pred[i] = prediction(dual, x_test[i,:])
    end
    errors = sum(abs.(y_test.-y_pred))/2
    print(errors, " errors out of ", length(x_test))

end

testSVM()
