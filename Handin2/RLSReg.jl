using ProximalOperators, LinearAlgebra

include("problem.jl")

function solve_RLS(x,y)
    w_rls = inv(transpose(X)*X + 2*lambda)*transpose(X)*y
