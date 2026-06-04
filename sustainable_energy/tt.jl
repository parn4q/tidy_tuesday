#=
This week we're exploring Sustainable Energy for all! Beyond the raw metrics, this dataset offers a window into how 
nations are balancing growth with green initiatives, challenging us to visualize the actual momentum behind the global
energy transition.

The “Sustainable Energy for all (SE4ALL)” initiative, launched in 2010 by the UN Secretary General, established three
global objectives to be accomplished by 2030: to ensure universal access to modern energy services, to double the 
global rate of improvement in global energy efficiency, and to double the share of renewable energy in the global 
energy mix. SE4ALL database supports this initiative and provides country level historical data for access to 
electricity and non-solid fuel; share of renewable energy in total final energy consumption by technology; 
and energy intensity rate of improvement.

Some questions to get you going:

- Which countries have the lowest capacity for solar energy?
- What form of renewable energy has, on average, experienced the fasted rate of adoption?

Thank you to Ntobeko Sosibo, Data Analyst and CG Hobbyist for curating this week's dataset.
=#


begin
    using DataFrames
    using Statistics
    using TidierTuesday
    using CairoMakie
    using AlgebraOfGraphics
    using CSV
end

begin
    data = tt_load("2026-05-26")
    df = data[1]
end

function safe_parse(x)
    # If it's already missing or is the string "NA", return missing
    if ismissing(x) || x == "NA"
        return missing
    else
        # Try to parse; if it's some other weird text, return missing instead of crashing
        return tryparse(Float64, string(x)) 
    end
end

transform!(
    df, 
    Between(:access_non_solid_fuel_rural_pop_pct, :wind_energy_consumption_terajoules) .=> 
    ByRow(safe_parse) .=> 
    Between(:access_non_solid_fuel_rural_pop_pct, :wind_energy_consumption_terajoules)
)


filter!(
    :country_name => !in(
        ["World", "Upper middle income", "Eastern Asia (including Japan)",
        "Eastern Asia (not including Japan)", "Eastern Europe", "Europe", 
        "High income", "High income: nonOECD", "High income: OECD", 
        "Low & middle income", "Low income", "Lower middle income", 
        "Middle income", "Northern Africa", "Northern America", "Oceania", 
        "Oceania (not including Australia and New Zealand)", 
        "South Eastern Asia", "Southern Asia", "Sub-Saharan Africa", 
        "Western Asia", "Western Sahara", "Caucasus and Central Asia",
        "Central African Republic", "Latin America and Caribbean", "Nothern America"]), 
    df
)



df_sum = describe(df)


# --- Lowest Capacity of Energy by country and year --- #

energy_by_country_yr = combine(
    groupby(
        df, [:country_name, :yr]
    ),
    :solar_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_solar,
    :biogas_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_biogas,
    :geothermal_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_geo,
    :hydro_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_hydro,
    :liquid_biofuels_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_biofuels,
    :marine_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_marine,
    :modern_biomass_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_modern_bio,
    :renewable_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_renewable,
    :traditional_biomass_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_trad_bio_mass,
    :waste_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_waste_energy,
    :wind_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_wind_energy
)


CSV.write("./to_interactive.csv", energy_by_country_yr)

########################################################################################################################

#=

# --- Lowest Capacity of Solar Energy --- #

se_by_country = combine(
    groupby(
        df, [:country_name]
    ),
    :solar_energy_consumption_terajoules => (x -> sum(skipmissing(x))) => :total_solar
)

filter!(x -> x.total_solar .> 0, se_by_country)

sort!(se_by_country, :total_solar, rev = true)

mean_solar = mean(se_by_country.total_solar)
median_solar = median(se_by_country.total_solar)


begin
    f = Figure(size = (1000, 600))

    ax = Axis(
        f[1,1], 
        xticks = (1:62, unique(se_by_country.country_name)),
        xticklabelrotation = 45,
        ytickformat = values -> map(v -> begin
            if v >= 1e6
                string(round(v / 1e6, digits=1), "M")
            elseif v >= 1e3
                string(round(v / 1e3, digits=1), "K")
            else
                string(round(v, digits=1))
            end
        end, values),
        limits = (0, 62.5, nothing, nothing),
        title = "Countries that have used the Most Solar Energy (in terajoules) over 20 Years")

    bp = AlgebraOfGraphics.data(se_by_country) * 
        mapping(:country_name => presorted, :total_solar) * 
        visual(BarPlot)

    draw!(ax, bp)

    hlines!(ax, mean_solar, color = :red, label = "Mean")
    hlines!(ax, median_solar, color = :blue, label = "Median")
    axislegend(ax, position = :rt)

    f
end
=#