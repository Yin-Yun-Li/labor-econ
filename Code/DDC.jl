using LinearAlgebra, Statistics, Distributions, Expectations, StatsBase
using LaTeXStrings, Plots, StatsPlots
using NLsolve, Roots, Random, Interpolations, Optim, ForwardDiff
using Optim: maximum, maximizer
using DataFrames
using StatsBase: Weights
using ShiftedArrays:lag
using ShiftedArrays:lead
using Chain
using DataFramesMeta


Random.seed!(12345)


### model_setup
function model_setup(; β::Vector=[2.5, 2.0], δ=0.95, C=0.8) 
    s = collect(0:12) ## post compulsory education: following age 12
    x = collect(0:12)  ## experience
    return (; β, δ, C, s, x)
end




### Bellman operator
function Q( EV, model)
    (; β, δ, C, s, x) = model

    T_EV = similar(EV)
    
    for (i, s_val) in enumerate(s)
        for (j, x_val) in enumerate(x)
            X = [s_val, x_val]
            u1 =   exp(log(X'β) + δ*EV[i,min(j+1, 13)])
            u0 =   exp(-C*x_val + δ*EV[min(i+1,13), j])
            T_EV[i,j] = log(u1+u0) 
        end
    end
    return T_EV
end    

### Value Function Iteration
function VFI(v_init, model; iterations = 500, m = 1, show_trace = true)
    res = fixedpoint(EV -> Q(EV, model), v_init; iterations, m, show_trace) # Anderson iteration
    bar_V_star = res.zero
    return (; bar_V_star,res=res)   
end  

### ccp
function cond_prob(a, vbar, model)
    prob_matrix = zeros(length(model.s), length(model.x))
    (; β, δ, C, s, x) = model

     for (i, s_val) in enumerate(s)
        for (j, x_val) in enumerate(x)
            X = [ s_val, x_val ]
            v1 =   log(X'β) + δ*vbar[ min(s_val, maximum(s)) + 1, min(x_val+1, maximum(x)) + 1]
            v0 =   -C*x_val  + δ*vbar[ min(s_val+1, maximum(s)) + 1, min(x_val, maximum(x)) + 1]

            p1 = exp(v1)/(exp(v1)+exp(v0))
            p0 = exp(v0)/(exp(v1)+exp(v0))

            prob_matrix[i,j] = a == 1 ? p1 : p0
        end
    end
    return prob_matrix
end    


#=
model = model_setup(β=[100, 2.5], δ=0.95,C=0.5)
v_init = zeros(length(model.s), length(model.x))
sol = VFI(v_init, model)
sol.bar_V_star

pr0 = cond_prob(0, sol.bar_V_star, model)
pr1 = cond_prob(1, sol.bar_V_star, model)
pr1 + pr0 
=#


#### paras:  β::Vector
### We only care about β=[β1, β2]; C is normalize to 0 and set calibrated δ=0.95 
function simulation_dt( δ=0.95, n_periods=13; n_agents=1000,  β::Vector=[2.5, 2.0], C = 0.8)

    if any(β .< 0)
        throw(ArgumentError("Elements of β must be non-negative."))
    end
    
    model = model_setup(β=β, δ=δ, C=C)
    v_init = zeros(length(model.s), length(model.x))
    sol = VFI(v_init, model)
    pr1 = cond_prob(1, sol.bar_V_star, model)

    dt = DataFrame(id=Int[], period=Int[], educ=Int[], experience=Int[], choice=Int[])

    for i in 1:n_agents
        s_val, x_val = 0, 0 ### initial condition  
        for t in 1:n_periods
            a = rand(Bernoulli(pr1[s_val+1, x_val+1]))            
            if a == 1
               s_prime, x_prime  = min(s_val, maximum(model.s)), min(x_val+1, maximum(model.x)) 
            else
               s_prime, x_prime  = min(s_val+1, maximum(model.s)), min(x_val, maximum(model.x)) 
            end
            # Append row to DataFrame
            push!(dt, (i, t, s_val, x_val, a))
            s_val, x_val = s_prime, x_prime
        end
    end

    return dt
end



dt = simulation_dt(n_agents=30000, β=[3.6, 1.79], C=2.58)
describe(dt)
vscodedisplay(dt)


#= ******************* Simulate the Data ***************
### 1. Plug numbers of agents and [β1, β2] into the simulation function (Non-negative).
### 2. Hopefully, both β1 and β2 do not exceed 20 , this could help me identify the parameters closely.
### 3. Larger sample helps a lot (n_agents ≥ 30000).

dt = simulation_dt(n_agents= , β=[ , ])

=#





function NFXP_innerloop(θ, dt, model::Function)
    
    (log_β1 , log_β2, log_C)=θ
    beta_1 = exp(log_β1)
    beta_2 = exp(log_β2)
    C = exp(log_C)
   # delta =  1/(1+exp(-d))
    inner_loop_model = model(β=[beta_1, beta_2], δ=0.95, C=C)
    inner_loop_model

    v_init = zeros(length(inner_loop_model.s), length(inner_loop_model.x))
    sol = VFI(v_init, inner_loop_model)


    p1 = cond_prob(1, sol.bar_V_star, inner_loop_model)
    p0 = cond_prob(0, sol.bar_V_star, inner_loop_model)

    log_lik=0.0
    Pi=0.0
    for (choice, educ, experience) in zip(dt.choice, dt.educ, dt.experience)
        Pi = choice == 0 ? p0[educ+1, experience+1] : p1[educ+1, experience+1]
        log_lik += log(Pi)
    end
    return -log_lik
end

## Check: NFXP_innerloop([0.0,0.0], dt, model_setup)
NFXP_innerloop([0.0,0.0,0.0], dt, model_setup)


function NFXP_outerloop(init::Vector, dt, model::Function)
    result = optimize(θ -> NFXP_innerloop(θ, dt, model), init, NelderMead())
    return β1_est, β2_est, C_est = exp(result.minimizer[1]), exp(result.minimizer[2]), exp(result.minimizer[3])
end

NFXP_outerloop([log(1.5), log(1.0), log(1.0)], dt, model_setup)


