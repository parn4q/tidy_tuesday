################
# Tidy Tuesday #
################


#=
The datasets this week come from the paper "Sex Ratio Bias Triggers 
Demographic Suicide in a Dense Tortoise Population". This topic seemed so 
interesting that even media like the New York Times picked it up: "Constant 
Sexual Aggression Drives Female Tortoises to Walk Off Cliffs".

In an exceptionally dense island population of Hermann's tortoises in Lake 
Prespa in North Macedonia, sexually coercive males dramatically overnumber 
females, inflict severe copulatory injuries and put them at risk of fatal 
falls from the island plateau's sheer rock faces. Harassed females are emaciated, 
reproduce less frequently, produce smaller clutches and have lower annual survival 
rates compared to females from a neighbouring mainland population. 
Sixteen years of capture-recapture data reveal an ongoing extinction event and 
predict that the last island female will die in 2083.

Questions: 
1. Do recaptures happen more often in spring or summer?
2. Does it seem easier to recapture male or female tortoises?
3. What are the differences among tortoises from the mainland vs the 
ones from the island in terms of body mass or carapace length?

Thank you to Novica Nakov, Free Software Macedonia for curating this week's dataset.
=#

begin
    using DataFrames
    import CSV: read
    using Statistics
    using StatsPlots
    using TidierTuesday
    using Random, LinearAlgebra, Distributions
    using ProgressBars
end

begin
    data = tt_load("2026-03-03")

    clutch = data[1]
    tort = data[2]
end

y = tort.body_mass_grams

sex = [d == "m" ? 1 : 0 for d in tort.sex]
season = [d == "Summer" ? 1 : 0 for d in tort.season]

locality = []

for i in 1:length(tort.locality)
    if tort.locality[i] == "Plateau"
        push!(locality, 0)
    elseif tort.locality[i] == "Beach"
        push!(locality, 1)
    else push!(locality, 2)
    end
end

X = hcat(ones(size(tort, 1)), 
    tort.body_condition_index,
    season,
    locality,
    sex .* season,
    tort.year_recode .* sex
)


# 1. Identify all unique tortoises
unique_ids = unique(tort.individual)
J = length(unique_ids)

# 2. Create a dictionary to map Names -> Numbers (1 to J)
id_map = Dict(id => i for (i, id) in enumerate(unique_ids))

# 3. Create 'turtle_idx': A vector the same length as your data, 
#    containing the integer ID for each row.
turtle_idx = [id_map[id] for id in tort.individual]

# 4. Create 'turtle_indices': A list of row numbers for each unique turtle.
#    (This is the fast lookup we discussed)
turtle_indices = [Int[] for _ in 1:J]
for (row_idx, t_id) in enumerate(turtle_idx)
    push!(turtle_indices[t_id], row_idx)
end

N = size(tort, 1)  # Or N = length(y)
K = size(X, 2)  # This looks at the number of columns in X
# Sort your data by individual and year before this!
# We need to know which observation is the "previous" one for each row
prev_idx = zeros(Int, N) # 0 if it's the first time seeing this turtle
deltas = zeros(N)

for idx_list in turtle_indices
    for i in 2:length(idx_list)
        curr = idx_list[i]
        prev = idx_list[i-1]
        prev_idx[curr] = prev
        deltas[curr] = tort.year_recode[curr] - tort.year_recode[prev]
    end
end

function log_posterior_phi(phi, u, sigma2_u, prev_idx, deltas)
    # Boundary check: phi must be between 0 and 1 for biological persistence
    if !(0.0 < phi < 0.99) return -Inf end
    
    lp = 0.0
    for i in 1:length(u)
        if prev_idx[i] != 0
            dt = deltas[i]
            mu = (phi^dt) * u[prev_idx[i]]
            # Variance scales with time; 1e-6 added for stability
            var = sigma2_u * (1.0 - phi^(2*dt)) + 1e-6 
            lp += -0.5 * (log(var) + (u[i] - mu)^2 / var)
        else
            # Initial captures are just N(0, sigma2_u)
            lp += -0.5 * (log(sigma2_u) + u[i]^2 / sigma2_u)
        end
    end
    return lp
end


# Initializing

# RESET TO REALITY
β = zeros(K)
β[1] = mean(y)   # Force the intercept to start at ~1500-2000
u = zeros(N)     # Force all random effects to start at 0
phi = 0.5        # Start with moderate autocorrelation
σ2_e = var(y) * 0.5
σ2_u = var(y) * 0.5


phi = 0.5
tuning = 0.07 # Adjust this if acceptance rate is too high/low
acceptances = 0
iterations = 2000

XtX = X'X

# Initialize with a fixed size and type
chain_phi = zeros(iterations)
# Storage for variances: Column 1 = σ2_error, Column 2 = σ2_innovation
chain_σ2 = zeros(iterations, 2)

chain_β = zeros(iterations, K)



for s in ProgressBar(1:iterations)
    # --- STEP 1: Update Fixed Effects (β) ---
    y_star = y - u 
    V_β = inv(XtX / σ2_e + I*1e-6)
    M_β = V_β * (X' * y_star / σ2_e)
    β = M_β + cholesky(Hermitian(V_β)).L * randn(K)

    # --- STEP 2: Update Autocorrelated Random Effects (u) ---
    # This is now a row-by-row update
    residuals = y - X * β
    for i in 1:N
        # This is a simplification; in a full state-space model 
        # u[i] depends on both the previous AND the next observation.
        # For hand-coding, a conditional mean works:
        prec = 1/σ2_e + 1/σ2_u
        m = (residuals[i]/σ2_e) / prec
        u[i] = m + sqrt(1/prec) * randn()
    end

    # --- STEP 3: Metropolis-Hastings for Phi ---
    phi_prop = phi + tuning * randn()
    l_ratio = log_posterior_phi(phi_prop, u, σ2_u, prev_idx, deltas) - 
              log_posterior_phi(phi, u, σ2_u, prev_idx, deltas)
    
    if log(rand()) < l_ratio
        phi = phi_prop
        acceptances += 1
    end

    # --- STEP 4: Update Variances (Inverse Gamma) ---
    # (Same as before, using residuals for e and u for u)
    # Residual error (Observation noise)
    err_e = y - X * β - u
    σ2_e = 1.0 / rand(Gamma(N/2, 1.0 / (sum(err_e.^2) / 2.0)))

    # State innovation (Process noise)
    # We only count the 'shocks'—the difference between u_t and its predicted value
    shocks = 0.0
    for i in 1:N
        if prev_idx[i] != 0
            prediction = (phi^deltas[i]) * u[prev_idx[i]]
            shocks += (u[i] - prediction)^2
        else
            shocks += u[i]^2 # Initial state variance
        end
    end

    σ2_u = 1.0 / rand(Gamma(N/2, 1.0 / (shocks / 2.0)))
    chain_β[s, :] = β
    chain_phi[s] = phi
    chain_σ2[s, :] = [σ2_e, σ2_u]
end

println(acceptances/iterations)


# Combine all parameters into one DataFrame
# Beta columns, Sigma2_e, Sigma2_u, and Phi
posterior_df = DataFrame(
    b0 = chain_β[:, 1],
    b_bci = chain_β[:, 2], 
    b_season = chain_β[:, 3],
    b_locality = chain_β[:, 4],
    b_interss = chain_β[:, 5],
    b_intersy = chain_β[:, 6], 
    sigma2_error = chain_σ2[:, 1],
    sigma2_turtle = chain_σ2[:, 2],
    phi = chain_phi
)


# We loop through the column names (symbols) of the DataFrame
for col in names(posterior_df)
    data = posterior_df[!, col]
    
    m = mean(data)
    # 95% Credible Interval
    low = quantile(data, 0.025)
    high = quantile(data, 0.975)
    
    println("Parameter: $col")
    println("  Mean: $(round(m, digits=3))")
    println("  95% CI: [$(round(low, digits=3)), $(round(high, digits=3))]")
    println("-"^20)
end

# Skips the first 500 iterations (adjust as needed)
burn_in = 5
plot_range = burn_in:iterations

# Create a layout for all parameters
p1 = plot(plot_range, posterior_df.b0[plot_range], title="Intercept (b0)", ylabel="Value")
p4 = plot(plot_range, posterior_df.b_interss[plot_range], title="Interaction", ylabel="Value")
p5 = plot(plot_range, posterior_df.sigma2_error[plot_range], title="Error Var", ylabel="Value")
p6 = plot(plot_range, posterior_df.sigma2_turtle[plot_range], title="Turtle Var", ylabel="Value")
p7 = plot(plot_range, posterior_df.phi[plot_range], title="AR(1) Phi", ylabel="Value")

# Combine them into a single window
plot(p1, p2, p3, p4, p5, p6, p7, layout=(4, 2), size=(1000, 1000), legend=false)


# Get the rows for Turtle 1
rows = turtle_indices[1]

# Calculate Fixed + Random
total_pred = X[rows, :] * β + u[rows]

println("Predicted weights for Turtle 1 over time: ", total_pred)

plot_df = DataFrame(
    y = y[1:12],
    total_pred = total_pred
)

scatter(total_pred, tort[1:12, :body_mass_grams])