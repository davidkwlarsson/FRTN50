
using LinearAlgebra, Plots

include("functions.jl")
include("problem.jl")

Q,q,a,b = problem_data()

function dual_gradient(Q,q,a,b,gamma)
    ITER = 50000
    y_k = randn(length(a))
    res = zeros(ITER)
    L = norm(Q,2)
    y_k1 = similar(y_k)
    f_y = zeros(ITER)
    f_x = zeros(ITER)
    x_k = zeros(length(a),ITER)

    for i = 1:ITER
        y_k1 = - prox_boxconj( - (y_k - gamma*grad_quadconj(y_k,Q,q)), a, b, gamma)
        res[i] = norm(y_k1 - y_k,2)
        y_k = y_k1
        f_y[i] = quadconj(y_k,Q,q)
        x_k[:,i] = dual2primal(y_k,Q,q,a,b)
        f_x[i] = quad(x_k[:,i],Q,q)
    end

    return y_k,res,f_x
end

function proxi_gradient(Q,q,a,b, gamma, ITER)
    x_k = randn(length(a))
    res = zeros(ITER)
    x_k1 = similar(x_k)
    f_x = zeros(ITER)
    for i = 1:ITER
        y_k = x_k - gamma*grad_quad(x_k,Q,q)
        x_k1 = prox_box(y_k, a, b, gamma)
        res[i] = norm(x_k1 - x_k,2)
        x_k = x_k1
        f_x[i] = quad(x_k, Q, q)
        if i == 1
        end
    end
    return x_k,res,f_x
end

# the best gamma is 8/5L
function plot_prim(iter, gamma = LinRange(0,2/norm(Q,2),5))
    L = 2/norm(Q,2)
    #gamma = 8/(5*L)
    n_g = length(gamma)
    res = zeros(iter,n_g)
    x_star = zeros(length(a),n_g)
    f_x = zeros(iter,n_g)
    for i = 1:n_g
        x_star[:,i],res[:,i],f_x[:,i] = proxi_gradient(Q,q,a,b,gamma[i], iter)
    end
    plot(res,
    title = "Primal solution\n||x_k+1 - x_k||",
    label = ["gamma = 1/L" "gamma = 3/2L"],
    yaxis=(:log, (10^-20, 10)))
    savefig("res_prim.png")

    #print(res[end, :])

    #plot(x_star, ylims = (-1,1),
    #title = "The solutions x^* after 1000 iterations, gamma = 8/(5L)",
    #label = "Primal solution",
    #xaxis = "i",
    #yaxis = "x_i",
    #seriestype=:scatter)
end

# gamma = 8/5L is good here as well
function plot_conj(gamma = LinRange(0,2/norm(inv(Q),2),5))
    L = norm(inv(Q),2)
    n_g = length(gamma)
    res_conj = zeros(50000,n_g)
    f_x = zeros(50000, n_g)
    x_star = zeros(length(a),n_g)
    for i = 1:n_g
        y_star,res_conj[:,i],f_x[:,i] = dual_gradient(Q,q,a,b,gamma[i])
        x_star[:,i] = dual2primal(y_star,Q,q,a,b)
    end
    #plot(res_conj, title = "Dual solution\n ||x_k+1 - x_k||, gamma = 8/(5L)", label = "")
    #savefig("res_dual.png")
    plot(res_conj[10000:end, 2:end-1],
    title = "Dual solution\n||x_k+1 - x_k||, removing the 10000 first iterations",
    label = ["gamma = 1/2L" "gamma = 1/L" "gamma = 3/2L"],
    yaxis=log10)
    #savefig("res_dual.png")
    #print(res_conj[end, :])


    #plot!(x_star, ylims = (-1, 1),
    #label = "Dual solution",
    #seriestype=:scatter)
    #savefig("both_nearly_converged.png")
end
