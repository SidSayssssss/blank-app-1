import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import hashlib
import re
from fpdf import FPDF
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="🌎 Superposition Markets Ultimate", layout="wide", page_icon="🌍")

# --- Translation dict (minimal example, extend as you want) ---
T = {
    "buy_local": "Buy Local 🏘️",
    "buy_imported": "Buy Imported 🚢",
    "invest_green": "Invest in Green Energy 🌱",
    "invest_fossil": "Invest in Fossil Fuels ⛽",
    "save_more": "Save More 💰",
    "spend_more": "Spend More 💳",
    "carbon_tax": "Carbon Tax ($/ton)",
    "green_subsidy": "Green Subsidy (index)",
    "meat_tax": "Meat Tax ($/kg)",
    "equity_focus": "Equity Focus",
    "n_agents": "Number of Agents",
    "scenario_key": "Scenario Key",
    "load_scenario": "Load scenario by key",
    "simulation_complete": "Simulation complete!",
    "future_diary": "📝 Future Diary",
    "badges": "🏅 Badges & Leaderboard",
    "scenario_comparison": "🔎 Scenario Comparison",
    "map_chart_toggle": "Map or Chart metric",
}

# --- Sidebar: User info & personal choices ---
st.sidebar.header("👤 About You")
location = st.sidebar.text_input("Where are you from?", value="United States")
occupation = st.sidebar.selectbox("Occupation", ["Student", "Professional", "Policymaker", "Academic", "Other"])

st.sidebar.header("⚙️ Lifestyle Choices")
buy_option = st.sidebar.selectbox("Buy option", [T["buy_local"], T["buy_imported"]])
invest_option = st.sidebar.selectbox("Invest option", [T["invest_green"], T["invest_fossil"]])
save_option = st.sidebar.selectbox("Save option", [T["save_more"], T["spend_more"]])

st.sidebar.header("🏛️ Policy Sandbox")
carbon_tax = st.sidebar.slider("Carbon Tax ($/ton)", 0, 200, 50)
green_subsidy = st.sidebar.slider("Green Subsidy (index)", 0, 100, 50)
meat_tax = st.sidebar.slider("Meat Tax ($/kg)", 0, 20, 5)
equity_focus = st.sidebar.slider("Equity Focus", 0.0, 1.0, 0.5, step=0.05)

# --- Additive policy sliders ---
risk_aversion = st.sidebar.slider("Risk Aversion Factor", 0.0, 1.0, 0.5, step=0.05,
                                  help="Higher risk aversion slows down climate risk taking")

green_mandate = st.sidebar.checkbox("Green Mandate Enabled",
                                    help="Requires minimum green energy investment by agents")

carbon_capture_subsidy = st.sidebar.slider("Carbon Capture Subsidy ($/ton)", 0, 100, 20)

renewable_energy_quota = st.sidebar.slider("Renewable Energy Quota (%)", 0, 100, 40)

climate_migration_policy = st.sidebar.checkbox("Enable Climate Migration Policy",
                                              help="Simulate migration due to climate disasters")

st.sidebar.header("🔬 Simulation Settings")
n_agents = st.sidebar.slider(T["n_agents"], 5, 50, 10)

# --- Scenario sharing ---
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

current_scenario = {
    "buy": buy_option,
    "invest": invest_option,
    "save": save_option,
    "carbon_tax": carbon_tax,
    "green_subsidy": green_subsidy,
    "meat_tax": meat_tax,
    "equity_focus": equity_focus,
    "n_agents": n_agents,
    "risk_aversion": risk_aversion,
    "green_mandate": green_mandate,
    "carbon_capture_subsidy": carbon_capture_subsidy,
    "renewable_energy_quota": renewable_energy_quota,
    "climate_migration_policy": climate_migration_policy,
}

def generate_scenario_hash(scenario: dict) -> str:
    s = json.dumps(scenario, sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()[:8]

def save_scenario_to_session(scenario: dict):
    key = generate_scenario_hash(scenario)
    st.session_state.saved_scenarios[key] = scenario
    return key

def load_scenario_from_key(key: str):
    return st.session_state.saved_scenarios.get(key, None)

save_key = save_scenario_to_session(current_scenario)
st.sidebar.write(f"Scenario Key: `{save_key}`")
load_key = st.sidebar.text_input(T["load_scenario"])
if load_key:
    loaded = load_scenario_from_key(load_key)
    if loaded:
        st.success("Scenario loaded! Please manually update sliders to match it.")
    else:
        st.error("Key not found.")

# --- Data fetcher ---
@st.cache_data(ttl=3600)
def fetch_real_time_co2_data() -> pd.DataFrame:
    try:
        response = requests.get("https://api.vizhub.energy/api/co2_emissions")
        if response.status_code == 200:
            data_json = response.json()
            df = pd.DataFrame(data_json["countries"])
            if {"Country", "CO2_2023", "GDP_2023"}.issubset(df.columns):
                return df
    except:
        pass
    data = {
        "Country": ["United States", "China", "Germany", "Nigeria", "Brazil", "Australia"],
        "CO2_2023": [4800, 10000, 800, 250, 400, 350],
        "GDP_2023": [23000, 17000, 4200, 400, 1800, 1400],
    }
    return pd.DataFrame(data)

def simulate_agent_based_scenario(n_agents: int, user_profile: dict, scenario_params: dict, years: list = list(range(2023, 2041))) -> pd.DataFrame:
    agents_data = []
    for agent_id in range(n_agents):
        footprint_factor = user_profile.get("footprint_factor", 1.0)
        carbon_tax = scenario_params.get("carbon_tax", 50)
        subsidy = scenario_params.get("green_subsidy", 50)
        meat_tax = scenario_params.get("meat_tax", 5)
        equity_focus = scenario_params.get("equity_focus", 0.5)

        agent_carbon_tax = carbon_tax * np.clip(np.random.normal(1.0, 0.1), 0.8, 1.2)
        agent_subsidy = subsidy * np.clip(np.random.normal(1.0, 0.1), 0.8, 1.2)
        base_emissions = 5000 * footprint_factor - agent_id * 10 - agent_carbon_tax * 0.5 - agent_subsidy * 0.3
        emissions = max(base_emissions, 1000)

        social_happiness = max(min(1.0 - emissions / 10000 + equity_focus * 0.3 + np.random.normal(0,0.05), 1.0), 0)
        economic_inequality = max(min(0.3 + emissions / 15000 - equity_focus * 0.2 + np.random.normal(0,0.05), 1.0), 0)
        health_impact = max(min(0.5 + emissions / 20000 - subsidy * 0.005 + np.random.normal(0,0.05), 1.0), 0)

        disaster_events = ["none", "flood", "drought", "storm", "heatwave"]
        disaster_prob = min(0.05 + emissions / 20000, 0.3)
        event = np.random.choice(disaster_events, p=[1-disaster_prob, disaster_prob*0.25, disaster_prob*0.25, disaster_prob*0.25, disaster_prob*0.25])

        df_agent = pd.DataFrame({
            "Year": years,
            "AgentID": agent_id,
            "CO2 Emissions (Mt)": [max(emissions - y * 10, 500) for y in range(len(years))],
            "Social Happiness Index": [social_happiness + np.random.normal(0, 0.01) for _ in years],
            "Economic Inequality": [economic_inequality + np.random.normal(0, 0.01) for _ in years],
            "Health Impact Index": [health_impact + np.random.normal(0, 0.01) for _ in years],
            "Event": [event for _ in years],
        })
        agents_data.append(df_agent)
    return pd.concat(agents_data, ignore_index=True)

# --- PDF Diary ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Superposition Markets Sustainability Report', 0, 1, 'C')
        self.ln(5)
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(3)
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln()
def create_pdf_report(text: str) -> bytes:
    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title("Personalized Future Report")
    pdf.chapter_body(text)
    return pdf.output(dest='S').encode('latin1', errors='replace')

def generate_disaster_map(df_agents, year):
    country_coords = {
        "United States": [37.0902, -95.7129],
        "China": [35.8617, 104.1954],
        "Germany": [51.1657, 10.4515],
        "Nigeria": [9.0820, 8.6753],
        "Brazil": [-14.2350, -51.9253],
        "Australia": [-25.2744, 133.7751]
    }
    df_year = df_agents[df_agents["Year"] == year].copy()
    np.random.seed(42)
    df_year["Country"] = np.random.choice(list(country_coords.keys()), size=len(df_year))
    df_year["Latitude"] = df_year["Country"].map(lambda c: country_coords[c][0])
    df_year["Longitude"] = df_year["Country"].map(lambda c: country_coords[c][1])

    m = folium.Map(location=[20, 0], zoom_start=2)
    for _, row in df_year.iterrows():
        if row["Event"] != "none":
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=7,
                popup=f"Agent {row['AgentID']} Disaster: {row['Event'].capitalize()}",
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
            ).add_to(m)
    return m

# --- Tabs ---
tabs = st.tabs([
    "🌍 Global Data",
    "🔄 Advanced Agent-Based Simulation",
    T["scenario_comparison"],
    T["badges"],
    T["future_diary"],
])

# --- Tab: Global Data ---
with tabs[0]:
    st.header(f"🌍 Hello {location}!")

    co2_df = fetch_real_time_co2_data()
    map_metric = st.selectbox(T["map_chart_toggle"], ["CO2_2023", "GDP_2023"], index=0)
    fig = px.choropleth(co2_df, locations="Country", locationmode="country names", color=map_metric,
                        hover_name="Country", color_continuous_scale="Reds",
                        title=f"{map_metric} by Country (2023)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 CO₂ Emissions vs GDP Scatter Plot")
    fig2 = px.scatter(co2_df, x="GDP_2023", y="CO2_2023", hover_name="Country",
                      labels={"GDP_2023": "GDP per Capita (k$)", "CO2_2023": "CO₂ Emissions (Mt)"},
                      title="CO₂ Emissions vs GDP (2023)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Historical Global CO₂ Emissions (Mock Data)")
    years_hist = list(range(1990, 2024))
    emissions_hist = [35000 - (y - 1990) * 300 + np.random.randint(-500, 500) for y in years_hist]
    hist_df = pd.DataFrame({"Year": years_hist, "Global CO2 Emissions (Mt)": emissions_hist})
    fig3 = px.line(hist_df, x="Year", y="Global CO2 Emissions (Mt)", title="Global CO₂ Emissions Over Time")
    st.plotly_chart(fig3, use_container_width=True)

    quotes = [
        "“The Earth is what we all have in common.” — Wendell Berry",
        "“Climate change is the defining issue of our time.” — Barack Obama",
        "“In nature, nothing exists alone.” — Rachel Carson",
        "“We do not inherit the Earth from our ancestors; we borrow it from our children.” — Native American Proverb"
    ]
    st.info(np.random.choice(quotes))

# --- Tab: Advanced Simulation ---
with tabs[1]:
    user_profile = {"footprint_factor": 1.0}
    with st.spinner("Simulating agents... please wait."):
        df_agents = simulate_agent_based_scenario(n_agents, user_profile, current_scenario)
        st.session_state.df_agents = df_agents  # Save for other tabs
    st.success(T["simulation_complete"])

    display_mode_sim = st.radio("Display Simulation Data As:", ["Table", "Maps"], horizontal=True)

    if display_mode_sim == "Table":
        st.dataframe(df_agents.head(15))
    else:
        map_option = st.selectbox("Select map metric", [
            "Disaster Events",
            "CO2 Emissions (Mt)",
            "Social Happiness Index",
            "Economic Inequality",
            "Health Impact Index"
        ])

        sim_year = st.slider("Select Year", min_value=int(df_agents["Year"].min()), max_value=int(df_agents["Year"].max()), value=int(df_agents["Year"].max()))
        per_capita = st.checkbox("Per Capita View", value=False)

        if map_option == "Disaster Events":
            m = generate_disaster_map(df_agents, sim_year)
            st_folium(m, width=700, height=450)
        else:
            df_year = df_agents[df_agents["Year"] == sim_year].copy()
            np.random.seed(42)
            countries_list = ["United States", "China", "Germany", "Nigeria", "Brazil", "Australia"]
            df_year["Country"] = np.random.choice(countries_list, size=len(df_year))
            metric_column = map_option
            if per_capita:
                df_year[metric_column] = df_year[metric_column] / 1000

            agg = df_year.groupby("Country")[metric_column].mean().reset_index()
            fig = px.choropleth(
                agg,
                locations="Country",
                locationmode="country names",
                color=metric_column,
                hover_name="Country",
                color_continuous_scale="Viridis",
                title=f"{map_option} by Country in {sim_year} ({'Per Capita' if per_capita else 'Total'})"
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Tab: Scenario Comparison ---
with tabs[2]:
    st.header("🔎 Scenario Comparison")

    if not st.session_state.saved_scenarios:
        st.info("No saved scenarios available. Adjust sliders and save scenarios first.")
    else:
        selected_keys = st.multiselect("Select scenarios to compare", list(st.session_state.saved_scenarios.keys()), default=list(st.session_state.saved_scenarios.keys())[:2])

        if len(selected_keys) < 2:
            st.warning("Select at least two scenarios for comparison.")
        else:
            comparison_rows = []
            for key in selected_keys:
                scenario = st.session_state.saved_scenarios[key]
                sim_df = simulate_agent_based_scenario(n_agents, user_profile, scenario)
                avg_emissions = sim_df.groupby("Year")["CO2 Emissions (Mt)"].mean().iloc[-1]

                comparison_rows.append({
                    "Scenario Key": key,
                    "Buy Option": scenario["buy"],
                    "Invest Option": scenario["invest"],
                    "Carbon Tax ($/ton)": scenario["carbon_tax"],
                    "Green Subsidy": scenario["green_subsidy"],
                    "Meat Tax ($/kg)": scenario["meat_tax"],
                    "Equity Focus": scenario["equity_focus"],
                    "Agents": scenario["n_agents"],
                    "Final Year Avg Emissions (Mt)": round(avg_emissions, 1)
                })

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df)

            fig = px.line()
            for key in selected_keys:
                scenario = st.session_state.saved_scenarios[key]
                sim_df = simulate_agent_based_scenario(n_agents, user_profile, scenario)
                yearly_avg = sim_df.groupby("Year")["CO2 Emissions (Mt)"].mean().reset_index()
                fig.add_scatter(x=yearly_avg["Year"], y=yearly_avg["CO2 Emissions (Mt)"], mode='lines', name=f"Scenario {key}")

            fig.update_layout(title="Average CO₂ Emissions Over Time per Scenario", yaxis_title="CO₂ Emissions (Mt)")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab: Badges & Leaderboard ---
with tabs[3]:
    st.header("🏅 Badges & Leaderboard")

    if "df_agents" not in st.session_state:
        st.info("Run the simulation in the Advanced Agent-Based Simulation tab to generate agent data first.")
    else:
        df_agents = st.session_state.df_agents

        total_emissions = df_agents.groupby("AgentID")["CO2 Emissions (Mt)"].sum().reset_index()
        total_emissions = total_emissions.sort_values(by="CO2 Emissions (Mt)")

        def badge_for_emissions(emission):
            if emission < 15000:
                return "🌟 Climate Champion"
            elif emission < 30000:
                return "👍 Eco Advocate"
            else:
                return "⚠️ Needs Improvement"

        total_emissions["Badge"] = total_emissions["CO2 Emissions (Mt)"].apply(badge_for_emissions)

        st.subheader("Leaderboard: Lowest Total CO₂ Emissions")
        st.dataframe(total_emissions)

        badge_counts = total_emissions["Badge"].value_counts()
        st.markdown("### Badge Summary")
        for badge, count in badge_counts.items():
            st.markdown(f"- {badge}: {count} agents")

# --- Tab: Future Diary ---
with tabs[4]:
    st.header("📝 Future Diary")

    if "df_agents" not in st.session_state:
        st.info("Run the simulation in the Advanced Agent-Based Simulation tab to generate future data first.")
    else:
        df_agents = st.session_state.df_agents

        years = sorted(df_agents["Year"].unique())
        year = st.slider("Select Year to view diary entry", min(years), max(years), max(years))

        df_year = df_agents[df_agents["Year"] == year]

        avg_emissions = df_year["CO2 Emissions (Mt)"].mean()
        avg_happiness = df_year["Social Happiness Index"].mean()
        avg_inequality = df_year["Economic Inequality"].mean()

        st.markdown(f"### Year: {year}")
        st.markdown(f"- Average CO₂ Emissions: {avg_emissions:.1f} Mt")
        st.markdown(f"- Average Social Happiness Index: {avg_happiness:.2f}")
        st.markdown(f"- Average Economic Inequality: {avg_inequality:.2f}")

        disasters = df_year[df_year["Event"] != "none"]["Event"].value_counts()
        if len(disasters) > 0:
            st.markdown("**Disaster Events Reported:**")
            for event, count in disasters.items():
                st.markdown(f"- {event.capitalize()}: {count} agents affected")
        else:
            st.markdown("No major disaster events reported this year.")

        if avg_emissions > 4000:
            st.warning("High CO₂ emissions are persisting, indicating limited climate mitigation.")
        elif avg_emissions < 1500:
            st.success("Emissions are low, reflecting strong climate action.")
