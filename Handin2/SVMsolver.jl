using ProximalOperators, LinearAlgebra, Plots


include("problem.jl")

x_train, y_train = svm_train()
x_test, y_test = svm_test_1()

function GaussKernel(x)
    N = length(x)
    K = zeros(N,N)
    sigma = 1
    for i = 1:N
        for j = 1:N
            K[i,j] = exp(-1/(2*sigma^2)*norm(x[i] - x[j], 2)^2)
        end
    end
    return K
end

function prediction(x,y,K)
    N = length(y)
    m = zeros(N)
    for i = 1 : N
        m_i = -1/2 * y[i].* transpose(x)*K[i,:]
        m[i] = sign(m_i[1])
    end
    return m
end

function prox_grad_method(x,y)
    lambda = 0.01
    ITER = 1
    K = GaussKernel(x)
    N = length(x)
    Q = diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),N))
    g = SqrNormL2(1/(2*lambda))
    w_k = randn(N)
    res = zeros(ITER)
    for i = 1:ITER
        grad_g = 1/lambda * Q * w_k
        y_k = w_k - gamma*grad_g
        w_k1, hw =  prox(h1, y_k, gamma)
        w_k = w_k1
        # m = prediction(w_k,y,K)
        # print(m .- y)
    end
    #m = prediction(w_k,y,K)
    return w_k,y,K
end


function testSVM()
    w_k, y, K = prox_grad_method(x_train, y_train)
    K = GaussKernel(x)
    m = prediction(w_k, y_train, K)
    errors = sum(abs.(y_test.-sign.(m)))
    print(errors)

end

testSVM()
