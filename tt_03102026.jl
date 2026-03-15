###############################
# Tidy Tuesday week 3-10-2026 #
###############################

#=
In an online quiz, created as an independent project by Adam Kucharski, 
over 5,000 participants compared pairs of probability phrases 
(e.g. “Which conveys a higher probability: Likely or Probable?”) and 
assigned numerical values (0–100%) to each of 19 phrases. 
The resulting data can be used to analyse how people interpret common probability phrases.

Which phrases do people most disagree on, in relation to the probability they represent?
Which demographic background is the most optimistic?
Does the order people are shown phrases in change their interpretation?

Thank you to Nicola Rennie for curating this week's dataset.
=#

begin
    using CSV
    using DataFrames
    using Statistics
    using StatsBase: countmap
    using StatsPlots
    using TidierTuesday
    using Random, LinearAlgebra, Distributions, SparseArrays
    using ProgressBars
    using MixedModels
    using GLM
end

begin
    data = tt_load("2026-03-10")

    abs_jud = data[1]
    pair_comp = data[2]
    resp_meta = data[3]
end

# Get complete cases for resp_meta dataset.  First turn NA to Missing
begin
    allowmissing!(resp_meta)
    
    for col in eachcol(resp_meta)
        replace!(col, "NA" => missing)
    end

    # Then drop the rows with missing values
    dropmissing!(resp_meta)
end


df = innerjoin(abs_jud, resp_meta, on = :response_id)


# Validate

df_sum = describe(df)

# --- Assign numbers to categories --- #

age_dict = Dict(
    "45-54" => 5,
    "25-34" => 3,
    "75+" => 8,
    "35-44" => 4,
    "55-64" => 6, 
    "65-74" => 7,
    "18-24" => 2,
    "Under 18" => 1
)

english_dict = Dict(
    "English is my first language" => 1,
    "English is not my first language but I am fluent" => 2,
    "English is not my first language and I am not fluent" => 3
)

educ_dict = Dict(
    "Bachelor" => 4,
    "Postgraduate" => 5,
    "High school" => 2,
    "Some college" => 3,
    "Less than high school" => 1
)

df.age_band = map(x -> age_dict[x], df.age_band)
df.english_background = map(x -> english_dict[x], df.english_background)
df.education_level = map(x -> educ_dict[x], df.education_level)

CSV.write("./dftoR.csv", df)

# --- Freq Mixed Models --- #



lmmod = lmm(
    @formula(probability ~ 1 + term + age_band  + (1 + term |response_id)), 
    df
)

pred = predict(lmmod)

mean((df.probability .- pred).^2)

lmod = lm(
    @formula(probability ~ 1 + term + age_band  + education_level), 
    df
)

aic(lmod)

loglikelihood(lmod)

lr = -2 *(loglikelihood(lmod) - -324232.7468)




# --- Explore the Data --- #

function make_dummy_matrix(x::AbstractVector)
    # Find unique levels in the data
    unique_levels = unique(x)
    # Get the number of observations (rows) and levels (columns)
    num_obs = length(x)
    num_levels = length(unique_levels)

    # Initialize a boolean matrix (efficient memory usage)
    m = Matrix{Bool}(undef, num_obs, num_levels)

    # Populate the matrix manually
    for i in eachindex(unique_levels)
        # For each level, check where it matches the original data
        @. m[:, i] = x == unique_levels[i]
    end
    
    # Dummy coding typically drops one reference level to avoid multicollinearity (the "dummy variable trap").
    # To implement standard dummy coding, return all columns except the first one.
    return m[:, 2:end]
end

dum_mat = make_dummy_matrix(df.term)

y = df.probability

X = hcat(ones(length(y)), dum_mat, df.age_band)

function create_Z(ids)
    unique_ids = unique(ids)
    id_map = Dict(id => i for (i, id) in enumerate(unique_ids))
    
    row_indices = 1:length(ids)
    col_indices = [id_map[id] for id in ids]
    vals = ones(Float64, length(ids))
    
    return sparse(row_indices, col_indices, vals)
end

Z = create_Z(df.response_id)


# --- Gibbs sampler for a Random effects Model --- #


using Base.Threads

function parallel_gibbs_random_slopes(y, X, Z; n_iter=2000)
    N, p = size(X)
    q_total = size(Z, 2)
    n_groups = 4426
    k = Int(q_total / n_groups) 

    # Pre-index Z for each subject to avoid slicing inside the loop
    # This is a huge speed boost for sparse matrices
    row_indices = [findall(!iszero, Z[:, (j-1)*k + 1]) for j in 1:n_groups]
    
    # Init
    β = zeros(p)
    u = zeros(q_total)
    σ2_ε = 10.0
    Σ_u = Matrix(1.0I, k, k)
    
    trace_β = zeros(n_iter, p)
    trace_σ2 = zeros(n_iter)
    trace_u = zeros(n_iter, q_total)
    trace_u_sd = Vector{Matrix{Float64}}(undef, n_iter)

    for t in ProgressBar(1:n_iter)
        # 1. Update β (Fixed Effects)
        resid_u = y - Z * u
        V_β_inv = (X' * X) / σ2_ε + I * 1e-6
        L_β = cholesky(Hermitian(V_β_inv))
        m_β = L_β \ (X' * resid_u / σ2_ε)
        β = m_β + L_β.U \ randn(p)

        # 2. Update u (Random Effects) - PARALLELIZED
        inv_Σ_u = inv(Σ_u)
        Xβ = X * β
        
        for j in 1:n_groups
            idx = ((j-1)*k + 1):(j*k)
            rows = row_indices[j]
            
            # Sub-matrices for subject j
            Zj = Z[rows, idx]
            yj = y[rows]
            Xβj = Xβ[rows]
            
            ZtZ_j = Zj' * Zj
            Ztr_j = Zj' * (yj - Xβj)
            
            prec_u_j = (ZtZ_j / σ2_ε) + inv_Σ_u
            L_u_j = cholesky(Hermitian(prec_u_j))
            m_u_j = L_u_j \ (Ztr_j / σ2_ε)
            
            # Update the shared u vector at the specific subject slice
            u[idx] = m_u_j + L_u_j.U \ randn(k)
        end

        # 3. Update σ2_ε
        resid = y - X*β - Z*u
        σ2_ε = 1/rand(Gamma(0.001 + N/2, 1/(0.001 + dot(resid, resid)/2)))

        # 4. Update Σ_u (Inverse Wishart)
        U_mat = reshape(u, k, n_groups)' 
        S = U_mat' * U_mat + I * 1e-4
        Σ_u = rand(InverseWishart(n_groups + k, S))

        trace_β[t, :] = β
        trace_σ2[t] = σ2_ε
        trace_u[t, :] = u
        trace_u_sd[t] = copy(Σ_u)
        
    end
    
    return (
        β=trace_β, 
        σ2=trace_σ2, 
        u_final=trace_u,
        Σ_final=trace_u_sd
        )
end

b, s, u, o = parallel_gibbs_random_slopes(y, X, Z)

beta_list = []

for n in 1:size(b, 2)
    p = plot(1:2000, b[:, n], title = n)
    push!(beta_list, p)
end

plot(beta_list..., size = (1000,1000))

plot(1:2000, s)

u_list = []
for n in 1:20
    p = plot(1:2000, u[:, n], title = n)
    push!(u_list, p)
end

plot(u_list..., size = (1000,1000))


plot(1:2000, o[1:2000, 1, 1])


# Use the last 500 iterations for the posterior mean
avg_Σ = mean(o[1500:2000])
# Convert Σ to a Correlation Matrix
std_devs = sqrt.(diag(avg_Σ))
corr_mat = avg_Σ ./ (std_devs * std_devs')

function calculate_log_likelihood(y, X, Z, β, u, σ2_ε)
    N = length(y)
    
    # 1. Calculate the residuals: (y - Xβ - Zu)
    # y is (N,), Xβ is (N,1), Zu is (N,1)
    resid = y .- (X * β) .- (Z * u)
    
    # 2. Sum of squared residuals
    ssr = dot(resid, resid)
    
    # 3. Log-likelihood formula
    # -N/2 * log(2πσ²) - (1/2σ²) * SSR
    ll = -0.5 * N * log(2 * π * σ2_ε) - (ssr / (2 * σ2_ε))
    
    return ll
end

# Assuming:
# b: trace_β (n_iter x p)
# s: trace_σ2 (n_iter)
# u: trace_u (n_iter x q_total)
# burn_in: 1000:2000

log_liks = [calculate_log_likelihood(y, X, Z, b[t, :], u[t, :], s[t]) for t in 500:2000]

mean_ll = mean(log_liks)
max_ll = maximum(log_liks) 

# Count parameters
n_fixed = size(X, 2)            # e.g., 21
n_resid_var = 1                 # σ2_ε
n_random_cov = (19 * 20) / 2    # 190 (for the 19x19 Σ_u matrix)

k = n_fixed + n_resid_var + n_random_cov
n_obs = length(y)

# BIC calculation using the Maximum Log-Likelihood from the chain
bic_val = k * log(n_obs) - 2 * max_ll

println("Bayesian BIC: ", bic_val)
