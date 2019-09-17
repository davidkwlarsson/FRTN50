"""
    quad(x,Q,q)

Compute the quadratic

	1/2 x'Qx + q'x

"""
function quad(x,Q,q)
	return 1/2 * transpose(x)*Q*x + transpose(q)*x
end



"""
    guadconj(y,Q,q)

Compute the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function quadconj(y,Q,q)
	retunr 1/2 transpose(y-q)*inv(Q)*(y-q)

end



"""
    box(x,a,b)

Compute the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end



"""
    boxconj(y,a,b)

Compute the convex conjugate of the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function boxconj(y,a,b)
	box_conj = similar(y)
	if(y[i] < 0)
		box_conj = a[i]*y[i]
	else
		box_conj = b[i]*y[i]
	return box_conj
end



"""
    grad_quad(x,Q,q)

Compute the gradient of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quad(x,Q,q)
	return Q*x .+ q
end



"""
    grad_quadconj(y,Q,q)

Compute the gradient of the convex conjugate of the quadratic

	1/2 x'Qx + q'x

"""
function grad_quadconj(y,Q,q)
	return inv(Q)*(y.-q)
end



"""
    prox_box(x,a,b)

Compute the proximal operator of the indicator function for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_box(x,a,b,gamma)
	
end



"""
    prox_boxconj(y,a,b)

Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function prox_boxconj(y,a,b,gamma)
	return y - prox_box(y,a,b,gamma)
end


"""
    dual2primal(y,Q,q,a,b)

Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y,Q,q,a,b)
	return grad_quadconj(y,Q,q)
end