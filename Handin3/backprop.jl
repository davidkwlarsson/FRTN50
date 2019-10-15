using LinearAlgebra, Statistics, Random

# We define some useful activation functions
sigmoid(x) = exp(x)/(1 + exp(x))
#+++ relu
relu(x) = max.(0, x)
#+++ leakyrelu
leakyrelu(x) = max.(x*0.2, x)

# And methods to calculate their derivatives
derivative(f::typeof(sigmoid), x::Float64) = sigmoid(x)*(1-sigmoid(x))
derivative(f::typeof(identity), x::Float64) = one(x)
#+++ derivative of relu
# p(x=0) = 0?
derivative(f::typeof(relu), x::Float64) = max(0, sign(x))
#+++ derivative of leakyrelu
derivative(f::typeof(leakyrelu), x::Float64) = max(-sign(x) * 0.2, sign(x))

# Astract type, all layers will be a subtype of `Layer`
abstract type Layer{T} end

""" Dense layer for `σ(W*z+b)`,
    stores the intermediary value z as well as the output, gradients and δ"""
struct Dense{T, F<:Function} <: Layer{T}
    W::Matrix{T}
    b::Vector{T}
    σ::F
    x::Vector{T}    # W*z+b
    out::Vector{T}  # σ(W*z+b)
    ∂W::Matrix{T}   # ∂J/dW
    ∇b::Vector{T}   # (∂J/db)ᵀ
    δ::Vector{T}    # dJ/dz
end



""" layer = Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1)
    Dense layer for `σ(W*x+b)` with nout outputs and nin inputs, with activation function σ.
    `W0, Wstd, b0, bstd` adjusts the mean and standard deviation of the initial weights. """
function Dense(nout, nin, σ::F=sigmoid, W0 = 1.0, Wstd = 0.1, b0=0.0, bstd = 0.1) where F
    W = W0/nin/nout .+ Wstd/nin/nout .* randn(nout, nin)
    b = b0 .+ bstd.*randn(nout)
    x = similar(b)
    out = similar(x)
    ∂W = similar(W)
    ∇b = similar(x)
    δ = similar(x, nin)
    Dense{Float64, F}(W, b, σ, x, out, ∂W, ∇b, δ)
end

""" out = l(z)
    Compute the output `out` from the layer.
    Store the input to the activation function in l.x and the output in l.out. """
function (l::Dense)(z)
    #+++ Implement the definition of a Dense layer here
    l.x .= l.W * z + l.b
    l.out .= l.σ.(l.x)
    return l.out
end

# A network is just a sequence of layers
struct Network{T,N<:Layer{T}}
    layers::Vector{N}
end

""" out = n(z)
    Comute the result of applying each layer in a network to the previous output. """
function (n::Network)(z)
    #+++ Implement evaluation of a network here
    input = z
    for l in n.layers
        input = l(input)
    end
    return input
end

""" δ = backprop!(l::Dense, δnext, zin)
    Assuming that layer `l` has been called with `zin`,
    calculate the l.δ = ∂L/∂zᵢ given δᵢ₊₁ and zᵢ,
    and save l.∂W = ∂L/∂Wᵢ and l.∇b = (∂L/∂bᵢ)ᵀ """
function backprop!(l::Dense, δnext, zin)
    #+++ Implement back-propagation of a dense layer here
    # Skipped the derivations but this should be the final expression
    # z_hat = l.x
    l.∇b .= δnext .* derivative.(l.σ, l.x)
    l.∂W .= l.∇b * transpose(zin)
    l.δ .= transpose(l.W) * l.∇b
    return l.δ
end


""" backprop!(n::Network, input, ∂J∂y)
    Assuming that network `n` has been called with `input`, i.e `y=n(input)`
    backpropagate and save all gradients in the network,
    where ∂J∂y is the gradient (∂J/∂y)ᵀ. """
function backprop!(n::Network, input, ∂J∂y)
    layers = n.layers
    # To the last layer, δᵢ₊₁ is ∂J∂y
    δ = ∂J∂y
    # Iterate through layers, starting at the end
    for i in length(layers):-1:2
        #+++ Fill in the missing code here
        zin = layers[i-1].out # Do we have access to this?
        δ = backprop!(layers[i], δ, zin)
    end
    # To first layer, the input was `input`
    zin = input
    δ = backprop!(layers[1], δ, zin)
    return
end



# This can be used to get a list of all parameters and gradients from a Dense layer
getparams(l::Dense) = ([l.W, l.b], [l.∂W, l.∇b])

""" `params, gradients = getparams(n::Network)`
    Return a list of references to all paramaters and corresponding gradients. """
function getparams(n::Network{T}) where T
    params = Array{T}[]         # List of references to vectors and matrices (arrays) of parameters
    gradients = Array{T}[]      # List of references to vectors and matrices (arrays) of gradients
    for layer in n.layers
        p, g = getparams(layer)
        append!(params, p)      # push the parameter references to params list
        append!(gradients, g)   # push the gradient references to gradients list
    end
    return params, gradients
end

### Define loss function L(y,yhat)
sumsquares(yhat,y) =  norm(yhat-y)^2
# And its gradient with respect to yhat: L_{yhat}(yhat,y)
derivative(::typeof(sumsquares), yhat, y) =  yhat - y

function gradientstep!(n, lossfunc, x, y)
    out = n(x)
    # Calculate (∂L/∂out)ᵀ
    ∇L = derivative(lossfunc, out, y)
    # Backward pass over network
    backprop!(n, x, ∇L)
    # Get list of all parameters and gradients
    parameters, gradients = getparams(n)
    # For each parameter, take gradient step
    for i = 1:length(parameters)
        p = parameters[i]
        g = gradients[i]
        # Update this parameter with a small step in negative gradient
        #→ direction
        p .= p .- 0.001.*g
        # The parameter p is either a W, or b so we broadcast to update all the
        #→ elements
    end
end

""" Structure for saving all the parameters and states needed for ADAM,
    as well as references to the parameters and gradients """
struct ADAMTrainer{T,GT}
    n::Network{T}
    β1::T
    β2::T
    ϵ::T
    γ::T
    params::GT              # List of paramaters in the network (all Wᵢ and bᵢ)
    gradients::GT           # List of gradients (all ∂Wᵢ and ∇bᵢ)
    ms::GT                  # List of mₜ for each parameter
    mhs::GT                 # List of \hat{m}ₜ for each parameter
    vs::GT                  # List of vₜ for each parameter
    vhs::GT                 # List of \hat{v}ₜ for each parameter
    t::Base.RefValue{Int}   # Reference to iteration counter
end

function ADAMTrainer(n::Network{T}, β1 = 0.9, β2 = 0.999, ϵ=1e-8, γ=0.1) where T
    params, gradients = getparams(n)
    ms = [zero(gi) for gi in gradients]
    mhs = [zero(gi) for gi in gradients]
    vs = [ones(size(gi)...) for gi in gradients]
    vhs = [zero(gi) for gi in gradients]
    ADAMTrainer{T, typeof(params)}(n, β1, β2, ϵ, γ, params, gradients, ms, mhs, vs, vhs, Ref(1))
end

""" `update!(At::ADAMTrainer)`
    Assuming that all gradients are already computed using backpropagation,
    take a step with the ADAM algorithm """
function update!(At::ADAMTrainer)
    # Get some of the variables that we need from the ADAMTrainer
    β1, β2, ϵ, γ = At.β1, At.β2, At.ϵ, At.γ
    # At.t is a reference, we get the value t like this
    t = At.t[]
    # For each of the W and b in the network
    for i in eachindex(At.params)
        p = At.params[i]        # This will reference either a W or b
        ∇p = At.gradients[i]    # This will reference either a ∂W or ∇b
        # Get each of the stored values m, mhat, v, vhat for this parameter
        m, mh, v, vh = At.ms[i], At.mhs[i], At.vs[i], At.vhs[i]

        # Update ADAM parameters and Take the ADAM step
        At.ms[i] = β1 * m + (1 + β1) * ∇p
        At.mhs[i] = At.ms[i] / (1 - β1^t)
        At.vs[i] = β2 * v + (1 + β2) * ∇p.^2
        At.vhs[i] = At.vs[i] / (1 - β2^t)
        At.params[i] .= p .- γ * At.mhs[i] ./ (sqrt.(At.vhs[i]) .+ ϵ)

    end
    At.t[] = t+1     # At.t is a reference, we update the value t like this
    return
end


""" `loss = train!(n, alg, xs, ys, lossfunc)`

    Train a network `n` with algorithm `alg` on inputs `xs`, expected outputs `ys`
    for loss-function `lossfunc` """
function train!(n, alg, xs, ys, lossfunc)
    lossall = 0.0           # This will keep track of the sum of the losses

    for i in eachindex(xs)  # For each data point
        xi = xs[i]          # Get data
        yi = ys[i]          # And expected output

        #+++ Do a forward and backwards pass
        #+++ with `xi`, `yi, and
        #+++ update parameters using `alg`
        out = n(xi)
        ∇L = derivative(lossfunc, out, yi)
        backprop!(n, xi, ∇L)
        update!(alg)

        loss = lossfunc(out, yi)
        lossall += loss
    end
    # Calculate and print avergae loss
    avgloss = lossall/length(xs)
    println("Avg loss: $avgloss")
    return avgloss
end

""" `testloss(n, xs, ys, lossfunc)`
    Evaluate mean loss of network `n`, over data `xs`, `ys`,
    using lossfunction `lossfunc` """
getloss(n, xs, ys, lossfunc) = mean(xy -> lossfunc(xy[2], n(xy[1])), zip(xs,ys))


#########################################################
#########################################################
#########################################################
### Task 3:

### Define network
# We use some reasonable value on initial weights
l1 = Dense(30, 1, leakyrelu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, leakyrelu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])

### This is the function we want to approximate
fsol(x) = [min(3,norm(x)^2)]

### Define data, in range [-4,4]
xs = [rand(1).*8 .- 4 for i = 1:2000]
# Task 5
# xs = [rand(1).*8 .- 4 for i = 1:30]
# no noise
ys = [fsol(xi) for xi in xs]
# noise
##ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]
# Test data
testxs = [rand(1).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)

### Train and plot
using Plots
# Train once over the data set
@time train!(n, adam, xs, ys, sumsquares)
scatter(xs, [copy(n(xi)) for xi in xs], label="",
        title="y-prediction after one epoch of training")

# Train 100 times over the data set
for i = 1:10
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

# Plot real line and prediction
plot(-4:0.01:4, [fsol.(xi)[1] for xi in -4:0.01:4], c=:blue,
    title="prediction after 1000 epochs of training", label="", ylims = (-0.2, 4))
scatter!(xs[1], ys[1], m=(:cross,0.2,:blue), label="y train")
scatter!(xs[2:end], ys[2:end], lab="", m=(:cross,0.2,:blue))
scatter!(xs[1], copy(n(xs[1])), m=(:circle,0.2,:red), label="y_pred")
scatter!(xs[2:end], [copy(n(xi)) for xi in xs[2:end]], m=(:circle,0.2,:red), label="")
# We can calculate the mean error over the training data like this also
getloss(n, xs, ys, sumsquares)
# Loss over test data like this
getloss(n, testxs, testys, sumsquares)
# 1 epochs: båda seten får ca 1 error
# 10 epochs: båda seten får ca 0.1 error
# 100 epochs: båda seten får ca e-5 error
# 1000 epochs: båda seten får ca e-5 error (liite bättre)
# Vi har typ konvergerat efter 100 itr.
# kan inte överträna då vår funktion är orealistiskt perfekt


# Plot expected line
plot(-8:0.01:8, [fsol.(xi)[1] for xi in -8:0.01:8], c=:blue,
        title="prediction after 1000 epochs of training", label="y true");
# Plot full network result
plot!(-8:0.01:8, [copy(n([xi]))[1] for xi in -8:0.01:8], c=:red,
    label="y pred", ylims = (-0.2, 4))
scatter!(xs[1], ys[1], m=(:cross,0.2,:blue), label="y train")
scatter!(xs[2:end], ys[2:end], lab="", m=(:cross,0.2,:blue))

#########################################################
#########################################################
#########################################################
### Task 4: GÖR ETT NYTT NÄTVÄRK
# Addera noise och träna om, vad händer? kan vi overfitta? (träna för noiset)
ys = [fsol(xi).+ 0.1.*randn(1) for xi in xs]

getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
# 1 epochs: båda seten får ca 1 error
# 10 epochs: train 0.01. test 0.004
# 100 epochs: train 0.01. test 0.0005

# Vi har typ konvergerat efter 100 itr.
#########################################################
#########################################################
#########################################################
### Task 5:
# Dra ner träningsdatan fårn 2000 till 30 och träna om (med fler iter)
# vad händer? vi borde få en imperfekt model då vi har bristfällig data.

getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
# få epochs ger inget vettigt.
# 1000 epochs: train 0.01, test 0.05
# 10000 epochs: train 0.002, test 0.008 typ optimalt tränad efter vår data
# 100000 epochs: train 0.0007 test 0.02. Vi har övertränat nätverket

#########################################################
#########################################################
#########################################################
### Task 6:
# återställ xs. ändra nätverket och träna om. sänk learning rate.
# vad händer? varför?

### New network
l1 = Dense(30, 2, relu, 0.0, 3.0, 0.0, 0.1)
lis = [Dense(30, 30, relu, 0.0, 3.0, 0.0, 0.1) for i = 1:4]
# Last layer has no activation function (identity)
ln = Dense(1, 30, identity, 0.0, 1.0, 0.0, 0.1)
n = Network([l1, lis..., ln])
### New function
fsol(x) = [min(0.5,sin(0.5*norm(x)^2))]

### x in R2
xs = [rand(2).*8 .- 4 for i = 1:2000]
ys = [fsol(xi) for xi in xs]
# Test data
testxs = [rand(2).*8 .- 4 for i = 1:1000]
testys = [fsol(xi) for xi in testxs]

### Define algorithm, last input is learning rate
adam = ADAMTrainer(n, 0.95, 0.999, 1e-8, 0.0001)
### Train and plot
using Plots

# Train 100 times over the data set
for i = 1:900
    # Random ordering of all the data
    Iperm = randperm(length(xs))
    @time train!(n, adam, xs[Iperm], ys[Iperm], sumsquares)
end

getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
# lr = 1e-4
# 100 epochs: 0.025, 0.031, 0.002 nu
# 1000 epochs: 0.00044, 0.00066
# lr = 1e-5
# 100 epochs: 0.32, 0.32
# 1000 epochs 0.0030, 0.0038
# Plotttnig that can be used for task 6:
scatter3d([xi[1] for xi in xs], [xi[2] for xi in xs], [n(xi)[1] for xi in xs], m=(:blue,1, :cross, stroke(0, 0.2, :blue)), size=(1200,800));
scatter3d!([xi[1] for xi in xs], [xi[2] for xi in xs], [yi[1] for yi in ys], m=(:red,1, :circle, stroke(0, 0.2, :red)), size=(1200,800))

########################################################
########################################################
########################################################
### Task 7:
# återställ learning rate ocg gör om t6 fast med relu ist för leaky.
# Vad händer? varför?
getloss(n, xs, ys, sumsquares)
getloss(n, testxs, testys, sumsquares)
# lr = 1e-4
# 100 epochs: 0.11, 0.11
# 1000 epochs:
