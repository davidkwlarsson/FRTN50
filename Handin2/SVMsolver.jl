using ProximalOperators, LinearAlgebra, Plots, Random, Statistics


include("problem.jl")


x_train, y_train = svm_train()
lambda = 0.001
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
    ITER = 100000
    K = GaussKernelMatrix(x)
    N = length(x)
    Q = 1/lambda * diagm(y)*K*diagm(y)
    gamma = 1/norm(Q,2)
    h1 = Conjugate(HingeLoss(ones(N),1/N))
    # g = SqrNormL2(1/(2*lambda))
    dual = randn(N)
    res = zeros(ITER)
    for i = 1:ITER
        grad_g = Q * dual
        y_k = dual - gamma*grad_g
        dual_1, hw =  prox(h1, y_k, gamma)
        res = norm(dual .- dual_1, 2)
        if i % 1000 == 0
            #println(res)
        end
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
    naive_class = 1
    #naive_class = -1
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
println("===============")
x_test, y_test = svm_test_2()
testSVM()
println("===============")
x_test, y_test = svm_test_3()
testSVM()
println("===============")
x_test, y_test = svm_test_4()
testSVM()
println("\n")

#NOTES : Testset2 seems best with (0.00001 , 0.25) and it classifies completely correct
#NOTEs : For the other testsets (0.001 , 0.5) seems to be the best
#NOTES : Testset 2 and 4 produces weird output for either.


x_data, y_data = svm_train()
index = randperm(500)
x_data = x_data[index]
y_data = y_data[index]
y_train = y_data[1:end-100]
x_train = x_data[1:end-100]
y_test = y_data[end-99:end]
x_test = x_data[end-99:end]

println("Running SVM using hold out with parameters: (lambda , gamma) = ("
            , lambda, " , " ,sigma ,") : ..." )
testSVM()  #11 18  11 18 9 11 9
println("Running SVM using k-fold with parameters: (lambda , gamma) = ("
            , lambda, " , " ,sigma ,") : ..." )

tot_error = 0
x_data, y_data = svm_train()
LENGTH = length(x_data)
batch = LENGTH/10
for k = 1:10
    index = randperm(LENGTH)
    global x_data = x_data[index]
    global y_data = y_data[index]
    global y_train = y_data[1:end-50]
    global x_train = x_data[1:end-50]
    y_test = y_data[end-49:end]
    x_test = x_data[end-49:end]
    k_error = testSVM()
    global tot_error += k_error
end

print("Average error over 10 iters = ", tot_error/10)
# 1.3 1.1 1.1 1.2 1.0
