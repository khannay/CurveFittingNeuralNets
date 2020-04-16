using Flux, Plots
using Base.Iterators: repeated
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using LinearAlgebra

gridsize = 150;
tstart=0.0
tend=20.0
myfunc(x) = sin(x)+0.5*sin(2*x)+0.24*cos(x);
x = collect(range(tstart,stop=tend,length=gridsize));
y= myfunc.(x);

xtest=collect(tend:0.1:2*tend)
xpred=collect(tstart:0.1:2*tend)


data = []
for i in 1:length(x)
    push!(data, ([x[i]], y[i])) #have to input a list
end

Q = 20;
ann = Chain(Dense(1,Q,sin),Dense(Q,1));


function loss(x, y)
    pred=ann(x)
    loss=Flux.mse(ann(x), y)
    return loss
end

opt = ADAM()

ps=params(ann)

for j=1:20
    ann[1].W[j]=j
end
delete!(ps, ann[1].W) #Make the first layers weights (frequencies not trainable)
@epochs 3000 Flux.train!(loss,ps, data, opt)


plot(xpred, ann(xpred')', color=:red, lw=2.0, label="")
scatter!(x,myfunc.(x), color=:blue, legend=false)
scatter!(xtest,myfunc.(xtest), color=:green, lw=2.0, label="", markershape=:cross)
ylims!((-1.5,1.5))
xlims!((tstart,2*tend))
title!("Curve Fitting with Neural Networks")
xlabel!("Input")
ylabel!("Output")
