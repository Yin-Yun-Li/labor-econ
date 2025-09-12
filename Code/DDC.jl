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
            u1 =   exp(log( 1 + X'β) + δ*EV[i,min(j+1, 13)])
            u0 =   exp(-C*x_val + δ*EV[min(i+1,13), j])
            T_EV[i,j] = log(u1+u0) 
        end
    end
    return T_EV
end    

### Value Function Iteration
function VFI(v_init, model ; iterations = 500, m = 1, show_trace = true, Tf::Function=Q)
    res = fixedpoint(EV -> Tf(EV, model), v_init; iterations, m, show_trace) # Anderson iteration
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
            v1 =   log( 1 + X'β) + δ*vbar[ min(s_val, maximum(s)) + 1, min(x_val+1, maximum(x)) + 1]
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
### Our goal is to recover β1, β2, C. Set calibrated δ=0.95. 
function simulation_dt( δ=0.95, n_periods=13; n_agents=1000,  β::Vector=[2.5, 2.0], C = 0.8, Tf::Function=Q, ccp = cond_prob)

    if any(β .< 0)
        throw(ArgumentError("Elements of β must be non-negative."))
    end
    
    model = model_setup(β=β, δ=δ, C=C)
    v_init = zeros(length(model.s), length(model.x))
    sol = VFI(v_init, model; Tf=Tf)
    pr1 = ccp(1, sol.bar_V_star, model)

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


#= ******************* Simulate the Data ***************
### 1. Plug numbers of agents, beta_vector and C into the simulation function (Non-negative).
### 2. Larger sample helps a lot (n_agents ≥ 30000).
=#

dt = simulation_dt(n_agents=30000, β=[3.6, 1.79], C=2.58)
describe(dt)
vscodedisplay(dt)

### Plotting the simulated data
decision_process = combine(groupby(dt, :period),
                           :choice => (x -> mean(x .== 0)) => :study_share,
                           :choice => (x -> mean(x .== 1)) => :work_share)

decision_process.age = decision_process.period .+ 17                           

p1 = plot(decision_process.age, decision_process.study_share,
     seriestype = :line,
     xlabel = "Age",
     ylabel = "Share of Decision",
     title = "",
     lw = 2,
     marker = :o,        
     color = :plum4,
     xticks = 18:30,
     label = "Study",
     legend = (0.90,0.25))

plot!(decision_process.age, decision_process.work_share,
     seriestype = :line,
     lw = 2,
     marker = :o,        
     color = :teal,
     label = "Work")

folder_path = raw"c:\\Users\\user\\Desktop\\github-local-repo\\labor-econ\\Report"
savefig(p1, joinpath(folder_path, "decision_by_age.png"))


### Estimation via NFXP
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

# Check: NFXP_innerloop([0.0,0.0, 0.0], dt, model_setup)


function NFXP_outerloop(init::Vector, dt, model::Function)
    result = optimize(θ -> NFXP_innerloop(θ, dt, model), init, NelderMead())
    return β1_est, β2_est, C_est = exp(result.minimizer[1]), exp(result.minimizer[2]), exp(result.minimizer[3])
end

est_res = NFXP_outerloop([log(1.5), log(1.0), log(1.0)], dt, model_setup)

println("estimated β1 is ", est_res[1])
println("estimated β2 is ", est_res[2])
println("estimated C is ", est_res[3])



### Policy Experiemnt
S = 1.5 # Subsidy on higher education 
est_model = model_setup( β=[est_res[1],est_res[2]],δ=0.95,C=est_res[3])

### Revisit Bellman operator
function T( EV, model)
    (; β, δ, C, s, x) = model

    T_EV = similar(EV)
    
    for (i, s_val) in enumerate(s)
        for (j, x_val) in enumerate(x)
            X = [s_val, x_val]
            u1 =   exp(log( 1 + X'β) + δ*EV[i,min(j+1, 13)])
            u0 =   exp( S -C*x_val + δ*EV[min(i+1,13), j])
            T_EV[i,j] = log(u1+u0) 
        end
    end
    return T_EV
end    


### modify ccp
function CCP(a, vbar, model)
    prob_matrix = zeros(length(model.s), length(model.x))
    (; β, δ, C, s, x) = model

     for (i, s_val) in enumerate(s)
        for (j, x_val) in enumerate(x)
            X = [ s_val, x_val ]
            v1 =   log( 1 + X'β) + δ*vbar[ min(s_val, maximum(s)) + 1, min(x_val+1, maximum(x)) + 1]
            v0 =   S - C*x_val  + δ*vbar[ min(s_val+1, maximum(s)) + 1, min(x_val, maximum(x)) + 1]

            p1 = exp(v1)/(exp(v1)+exp(v0))
            p0 = exp(v0)/(exp(v1)+exp(v0))

            prob_matrix[i,j] = a == 1 ? p1 : p0
        end
    end
    return prob_matrix
end    

counterfactual_dt = simulation_dt( n_agents=1000,  β=[est_res[1],est_res[2]], C=est_res[3], Tf=T, ccp=CCP)
describe(counterfactual_dt)


### Plot decision process comparison
decision_process_after_subsidy = combine(groupby(counterfactual_dt, :period),
                           :choice => (x -> mean(x .== 0)) => :study_share,
                           :choice => (x -> mean(x .== 1)) => :work_share)

decision_process_after_subsidy.age = decision_process.period .+ 17    

p2 = plot(decision_process.age, decision_process.study_share,
     seriestype = :line,
     xlabel = "Age",
     ylabel = "Share of Decision",
     title = "",
     lw = 2,
     marker = :o,        
     color = :plum4,
     xticks = 18:30,
     label = "Study",
     legend = (0.90,0.25))

plot!(decision_process.age, decision_process.work_share,
     seriestype = :line,
     lw = 2,
     marker = :o,        
     color = :teal,
     label = "Work")

plot!(decision_process_after_subsidy.age, decision_process_after_subsidy.study_share,
     linestyle = :dash,
     lw = 2,
     marker = :diamond,
     color = :purple,
     label = "Study (CF)")

plot!(decision_process_after_subsidy.age, decision_process_after_subsidy.work_share,
     linestyle = :dash,
     lw = 2,
     marker = :diamond,
     color = :slategrey,
     label = "Work (CF)",
     size=(800, 600))

savefig(p2, joinpath(folder_path, "decision_by_age_comparison.png"))


### Wage profile
dt.wage =  log.(1 .+ est_res[1] .*dt.educ .+ est_res[2] .*dt.experience)

dt_avg = combine(groupby(dt, :period),
                 :wage  => mean => :mean_wage)

counterfactual_dt.wage =  log.(1 .+ est_res[1] .*counterfactual_dt.educ .+ est_res[2] .*counterfactual_dt.experience)

counterfactual_dt_avg = combine(groupby(counterfactual_dt, :period),
                 :wage  => mean => :mean_wage)

dt_avg.age = dt_avg.period .+ 17    
counterfactual_dt_avg.age = counterfactual_dt_avg.period .+ 17 

p3 = plot(dt_avg.age, dt_avg.mean_wage,
     seriestype = :line,
     xlabel = "Age",
     ylabel = "Average Log Wage",
     title = "",
     lw = 2,
     marker = :o,        
     color = :plum4,
     xticks = 18:30,      
     label = "No Subsidy")    


plot!(counterfactual_dt_avg.age, counterfactual_dt_avg.mean_wage,
      seriestype = :line,
      lw = 2,
      marker = :diamond, 
      color = :lightsteelblue3,  
      label = "School Subsidy",
      size=(800, 600))

savefig(p3, joinpath(folder_path, "lifetime_utility.png"))






########### Additional Plotting
solveEV = VFI(zeros(length(est_model.s), length(est_model.x)), est_model; Tf=T)
pr1 = CCP(1, solveEV.bar_V_star, est_model)

# Create heatmap
ccp_plot = heatmap(
    est_model.x,                  # x-axis values
    est_model.s,                  # y-axis values
    pr1,                       # Matrix of values
    xlabel="Work Experiences", 
    ylabel="Years of Schooling", 
    title="Conditional Probability of Work",
    color=:plasma           # Adjust color scheme
)



