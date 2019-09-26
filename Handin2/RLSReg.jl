using ProximalOperators, LinearAlgebra, Plots


include("problem.jl")
x,y = leastsquares_data()

function prox_grad_method(x,y,q = 2, p = 1)
    Xt = ones(length(x))
    for i = 1:p
        Xt = cat(Xt, x.^i, dims = 2)
    end
    #print(size(Xt))
    gamma = 1/norm(transpose(Xt)*Xt, 2)
    lambda = 0.01
    ITER = 50000
    w_k = randn(p+1)
    if (q==1)
        g = NormL1(lambda)
    else
        g = SqrNormL2(lambda)
    end
    f = LeastSquares(Xt, y, 1.0; iterative = false)
    for i = 1:ITER

        grad_f, _ = gradient(f,w_k)
        y_k = w_k - gamma*grad_f
        w_k1, gw =  prox(g, y_k, gamma)
        w_k = w_k1
        #res[i] = norm(w_k1 - w_k, 2)
    end

    return w_k
end


function plot_model1(x,y)


    x1 = maximum(x)
    x2 = minimum(x)
    beta = (abs.(x1) - abs.(x2))/2
    sigma = x1-beta
    #x = (x .- beta) ./ sigma

    plot(x,y,seriestype=:scatter)
    xtemp = LinRange(minimum(x),maximum(x),100)
    p = 10
    m = zeros(length(xtemp),p)
    for i=10:p
        w = prox_grad_method(x,y,1,i)
        Xttemp = ones(length(xtemp))
        for j = 1:i
            Xttemp = cat(Xttemp, xtemp.^j, dims = 2)
        end
        m[:,i] = Xttemp*w

    end
    plot!(xtemp,m[:,end])

    #plot!(x,m[:,1:3])
    # print(length(m), "  :  ", length(x), "   :   ", length(y))
end


function solve_RLS(x,y)
    Xt = [ones(length(x)) x]
    lambda = 1
    w_rls = inv(Xt*transpose(Xt) + 2*lambda*identity())*Xt*y
    return w_rls
end
