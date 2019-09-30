using ProximalOperators, LinearAlgebra, Plots


include("problem.jl")
x_data, y_data = leastsquares_data()

# Solves the problem iterativly using the proximal gradient method
# Needs data x and y as input.
# Degree of feature map polynomial p optional, 1 defualt
# Regularization strategy q optional, L2-squared default
function prox_grad_method(x, y, p = 1, lambda = 0, ITER = 10^7, q = 2)
    Xt = ones(length(x))
    for i = 1:p
        Xt = cat(Xt, x.^i, dims = 2)
    end
    gamma = 1/norm(transpose(Xt) * Xt, 2)
    w_k = randn(p+1)
    if q == 1
        g = NormL1(lambda)
    else
        g = SqrNormL2(lambda)
    end
    f = LeastSquares(Xt, y, 1.0; iterative = false)
    res = zeros(ITER)
    res2 = zeros(ITER)
    for i = 1:ITER
        grad_f, _ = gradient(f, w_k)
        y_k = w_k - gamma * grad_f
        w_k1, gw =  prox(g, y_k, gamma)
        res[i] = norm(w_k1 - w_k, 2)
        res2[i] = norm(y - Xt * w_k1, 2)
        w_k = w_k1
    end
    return w_k, res
end

# Prescales x, plots original data, solves TP of choosen degree
# and then plots fitted line to the data.
function plot_model1(x, y, p_vec = 1, lambda = 0, ITER = 10^7, q = 2)
    # prescales x s.t. x_max = 1, x_min = -1
    x1 = maximum(x)
    x2 = minimum(x)
    beta = (abs.(x1) - abs.(x2))/2
    sigma = x1-beta
    #x = (x .- beta) ./ sigma

    plot(x,y,seriestype=:scatter, label="Data")
    x_grid = LinRange(x2,x1,100)

    model = zeros(length(x_grid), length(p_vec))
    res = zeros(ITER, length(p_vec))
    feature_map = ones(length(x_grid))
    for j = 1:p_vec[end]
        feature_map = cat(feature_map, x_grid.^j, dims = 2)
    end
    #for i = 1:length(p_vec)
    w, res[:, 1] = prox_grad_method(x, y, p_vec[1], lambda, ITER, q)
        # print("model size: ", size(model), "col: ", (p - p_start + 1), "\n",
            # "fm size: ", size(feature_map), "size w: ", size(w))
    model[:, 1] = feature_map[:, 1:p_vec[1]+1] * w
        # if p == p_max
        #     print(w)
        # end
    #end
    #plot!(x_grid, model[:, end])
    #return res
    #plot!(res[Int(Iter/10):end, :],
    #title = "log10(||w_k+1 w_k||) after 10^7 iterations \n removing the first 10 % of the iterations",
    #label = "lambda = 0.1",
    #yaxis=log10)

    #savefig("LS_err.png")

    plot!(x_grid, model,
    title = "Data and fitted model after 10^7 iterations, q=2, lambda = 0.1",
    label = "model without regularization"
    )
    #print!("w for lambda = ", lambda, ": ", w)
    #print("lambda: ", lambda)
    #print(" w: ", w, "\n")

    #savefig("LS_model.png")
    #label = ["p = 1" "p = 2" "p = 3" "p = 4" "p = 5" "p = 6" "p = 7" "p = 8" "p = 9" "p = 10"],


    #print(res[end,:])
    # plot!(x,m[:,1:3])
    # print(length(m), "  :  ", length(x), "   :   ", length(y))
end

# Classic RLS solver, not used
function solve_RLS(x,y)
    Xt = [ones(length(x)) x]
    lambda = 1
    w_rls = inv(Xt*transpose(Xt) + 2*lambda*identity())*Xt*y
    return w_rls
end

function snakeplot()
    x_grid = LinRange(-1.5,1.5,100)

    #L1 = NormL1(1)
    #L2sq = SqrNormL2(1)
    #plot(L1(x_grid))
    #plot!(L2sq(x_grid))
    plot(x_grid, abs.(x_grid),
    title="Regularization for term in RLS training problem for diferent q:s.\n lambda = 1",
    label="q=1",
    xaxis="omega")


end
snakeplot()
#savefig("LS_regterm.png")
# task2: lambda = 0, p = [1, 10]. q spelar ingen roll då lambda = 0
# task3: p = 10, q = 2. testa lambda = [0.001, 10]. Testa q = 1. convergens?
# task4: välj q och lambda från t3. ta bort regularisering av x. Vad händer?
p = 10
#p = [1,3,6,10]
#p = [1,2,3,4,5,6,7,8,9,10]
lambda = 10^-1
#lambda = [10^-3, 10^-1, 1, 10^1]
Iter = 10^7
q = 2
#print("\n")
res = zeros(Iter,1)
plot_model1(x_data, y_data, p, lambda, Iter, q)
savefig("LS_unreg.png")
#for i = 1:4
#    res[:,i] = plot_model1(x_data, y_data, p, lambda[i], Iter, q)
#end
plot(res[Int(Iter/10):Int(Iter/1), :], title = "log10(||w_k+1 w_k||) after 10^7 iterations \n removing the first 10 % of the iterations",label = ["lambda = 0.001" "lambda = 0.1" "lambda = 1" "lambda = 10"], yaxis=log10)

#savefig("LS_lambda_convq1.png")
#for i = 1:4
#    plot_model1(x_data, y_data, p, lambda[i], Iter, 2)
#end
#savefig("LS_lambda_q1.png")
