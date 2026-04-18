import pandas as pd
import numpy as np
import requests
from io import StringIO
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: Data Integration Pipeline
# =====================================================================

def fetch_eia_data(api_key="DEMO_KEY"):
    """
    Fetches 'Crude Oil Production' and 'Ending Stocks' from EIA API v2.
    """
    if api_key == "DEMO_KEY":
        print("[EIA] Using DEMO_KEY. Generating placeholder monthly data.")
        # Ensure we have timezone-naive dates up to present time
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        date_rng = pd.date_range(start='2010-01-01', end=end_date, freq='MS')
        df = pd.DataFrame(index=date_rng)
        # Random walk for supply and stocks
        df['eia_production'] = 1e7 + np.cumsum(np.random.normal(0, 50000, len(date_rng)))
        df['eia_ending_stocks'] = 1e6 + np.cumsum(np.random.normal(0, 10000, len(date_rng)))
        return df
    
    print(f"[EIA] Fetching data using API Key: {api_key}")
    # Example endponts for EIA v2 API
    prod_url = f"https://api.eia.gov/v2/petroleum/pnp/crq/data/?api_key={api_key}&frequency=monthly&data[0]=value"
    stock_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={api_key}&frequency=monthly&data[0]=value"
    
    try:
        print("[EIA] Fetching actual data from EIA API...")
        res_prod = requests.get(prod_url).json()
        df_prod = pd.DataFrame(res_prod['response']['data'])
        df_prod['date'] = pd.to_datetime(df_prod['period'])
        df_prod['value'] = pd.to_numeric(df_prod['value'], errors='coerce')
        # Aggregate to get total US/global proxy per month
        prod_series = df_prod.groupby('date')['value'].sum()
        
        res_stock = requests.get(stock_url).json()
        df_stock = pd.DataFrame(res_stock['response']['data'])
        df_stock['date'] = pd.to_datetime(df_stock['period'])
        df_stock['value'] = pd.to_numeric(df_stock['value'], errors='coerce')
        stock_series = df_stock.groupby('date')['value'].sum()
        
        df_eia = pd.DataFrame({
            'eia_production': prod_series,
            'eia_ending_stocks': stock_series
        })
        # Resample to MS (Month Start) to align perfectly with OWID
        df_eia = df_eia.resample('MS').mean()
        print(f"[EIA] Successfully loaded {len(df_eia)} rows from EIA API.")
        return df_eia
    except Exception as e:
        print(f"[EIA] Error fetching or parsing data: {e}")
        return pd.DataFrame()

def load_owid_data():
    """
    Loads 'Energy Consumption' and 'Fossil Fuel Consumption' for Thailand and Global
    from the given OWID CSV link.
    """
    url = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"
    print(f"[OWID] Fetching dataset from {url}...")
    try:
        # NOTE: This is a ~75MB file. In a Hackathon scenario this is fast enough.
        cols = ['country', 'year', 'primary_energy_consumption', 'fossil_fuel_consumption']
        df = pd.read_csv(url, usecols=cols, storage_options={'User-Agent': 'Mozilla/5.0'})
        
        # Filter for Thailand and World
        df_filtered = df[df['country'].isin(['Thailand', 'World'])]
        # Convert to datetime index
        df_filtered['date'] = pd.to_datetime(df_filtered['year'], format='%Y')
        df_filtered = df_filtered.set_index('date')
        
        # Pivot table to widen columns per country
        df_pivot = df_filtered.pivot_table(
            index=df_filtered.index, 
            columns='country', 
            values=['primary_energy_consumption', 'fossil_fuel_consumption']
        )
        # Flatten multi-index columns: e.g. ('primary_energy_consumption', 'World') -> 'primary_energy_consumption_World'
        df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
        
        return df_pivot
    except Exception as e:
        print(f"[OWID] Failed to load from URL. Error: {e}")
        print("[OWID] Generating fallback dummy yearly data.")
        date_rng = pd.date_range(start='2010-01-01', end='2024-01-01', freq='YS')
        df_pivot = pd.DataFrame(index=date_rng)
        df_pivot['primary_energy_consumption_World'] = 140000 + np.cumsum(np.random.normal(500, 1000, len(date_rng)))
        df_pivot['fossil_fuel_consumption_World'] = 110000 + np.cumsum(np.random.normal(300, 800, len(date_rng)))
        df_pivot['primary_energy_consumption_Thailand'] = 1000 + np.cumsum(np.random.normal(20, 50, len(date_rng)))
        return df_pivot

def resample_interpolate_owid(df_yearly):
    """
    Resamples and interpolates the yearly OWID data into monthly frequency
    to match EIA's monthly data.
    """
    print("[OWID] Resampling yearly to monthly frequency using Cubic Spline Interpolation...")
    df_yearly = df_yearly.sort_index()
    # Resample to Month Start
    df_monthly = df_yearly.resample('MS').mean()
    # Interpolate using spline, fallback to linear if it fails due to too few points
    df_monthly_interp = df_monthly.interpolate(method='cubicspline')
    if df_monthly_interp.isna().sum().sum() > 0:
        df_monthly_interp = df_monthly_interp.interpolate(method='linear')
    
    # Backfill and forward-fill the edges
    return df_monthly_interp.bfill().ffill()

def doeb_placeholder(index):
    """
    Placeholder for DOEB Thailand's monthly petroleum data.
    """
    print("[DOEB] Creating placeholder for Thailand's monthly petroleum data...")
    df = pd.DataFrame(index=index)
    # Base level around 100k + seasonality + noise
    seasonality = np.sin(np.arange(len(index)) / 12 * 2 * np.pi) * 10000
    df['doeb_demand_thailand'] = 100000 + seasonality + np.random.normal(0, 2000, len(index))
    return df

def build_dataset():
    eia_df = fetch_eia_data("DEMO_KEY")
    owid_yr = load_owid_data()
    owid_mo = resample_interpolate_owid(owid_yr)
    
    # Outer join to align timestamps perfectly
    df_merged = pd.concat([eia_df, owid_mo], axis=1).sort_index().dropna(how='all')
    
    # Crop to current date dynamically to make sure timeline reaches 2026+
    current_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    df_merged = df_merged.loc['2010-01-01':current_date_str]
    
    # Forward fill missing OWID data up to the present day
    df_merged = df_merged.ffill()
    
    # Merge DOEB Thailand data
    doeb_df = doeb_placeholder(df_merged.index)
    df_final = pd.concat([df_merged, doeb_df], axis=1)
    
    # Placeholder for Oil Price - Generating a synthetic oil price that spikes
    df_final['oil_price'] = 60 + np.cumsum(np.random.normal(0.2, 1.5, len(df_final)))
    
    # Final check for NaNs
    df_final = df_final.ffill().bfill()
    return df_final


# =====================================================================
# STEP 2: Feature Engineering (Divergence Logic)
# =====================================================================

def feature_engineering(df, country="Global"):
    print(f"[Feature Eng] Normalizing signals and calculating Systemic Tension Score for {country}...")
    df = df.copy()
    scaler = StandardScaler()
    
    # Assign our proxies
    # Supply = EIA Production
    # Inventory = EIA Ending Stocks
    supply_col = 'eia_production'
    inv_col = 'eia_ending_stocks'
    
    # Demand = World Energy Consumption or Thailand Energy Consumption
    if country == "Thailand":
        demand_col = 'primary_energy_consumption_Thailand'
    else:
        demand_col = 'primary_energy_consumption_World'
    
    # Normalize features using Z-score
    df['Supply_Z'] = scaler.fit_transform(df[[supply_col]])
    df['Demand_Z'] = scaler.fit_transform(df[[demand_col]])
    df['Inventory_Z'] = scaler.fit_transform(df[[inv_col]])
    df['Delta_Inventory_Z'] = df['Inventory_Z'].diff().fillna(0)
    
    # 🌟 Calculate Systemic Tension Score
    # Score = (Supply - Demand) - Delta Inventory
    df['Systemic_Tension_Score'] = (df['Supply_Z'] - df['Demand_Z']) - df['Delta_Inventory_Z']
    
    # 🌟 Rolling Correlation (e.g., 6 months window) between Score and Price
    print("[Feature Eng] Calculating rolling window correlation...")
    df['Rolling_Corr_Score_Price'] = df['Systemic_Tension_Score'].rolling(window=6).corr(df['oil_price']).fillna(0)
    
    # 🌟 Divergence Signal based on Euclidean distance proxy (Absolute Difference)
    distance = np.abs(df['Supply_Z'] - df['Demand_Z'])
    df['S_D_Distance'] = distance
    
    dist_mean = distance.mean()
    dist_std = distance.std()
    
    # Signal triggered when distance exceeds 2 standard deviations
    df['Divergence_Signal'] = np.where(distance > (dist_mean + 2 * dist_std), 1, 0)
    
    return df

# =====================================================================
# STEP 3: Probabilistic Modeling & Anomaly Detection
# =====================================================================

def probabilistic_modeling(df):
    print("[Modeling] Running Isolation Forest for Anomaly Detection...")
    df = df.copy()
    
    # Isolation Forest to detect anomalies in Tension Score
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    score_array = df['Systemic_Tension_Score'].fillna(0).values.reshape(-1, 1)
    
    # Predict returns -1 for outliers and 1 for inliers. Mapped to 1 (anomaly), 0 (normal)
    preds = iso_forest.fit_predict(score_array)
    df['IForest_Anomaly'] = np.where(preds == -1, 1, 0)
    
    print("[Modeling] Fitting Bayesian Structural Time Series (Unobserved Components)...")
    # Using statsmodels Local Level Model as a robust proxy for BSTS
    ts_data = df['Systemic_Tension_Score'].fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bsts_model = sm.tsa.UnobservedComponents(ts_data, level='local level')
        res = bsts_model.fit(disp=False)
    
    # Extract the true 'expected' equilibrium line via the smoothed states
    df['BSTS_Expected_Balance'] = res.smoothed_state[0]
    
    # Calculate Residuals (Actual - Expected)
    df['BSTS_Residuals'] = df['Systemic_Tension_Score'] - df['BSTS_Expected_Balance']
    
    # Significant Deviations in Residuals (Threshold > 1.5 Std Dev)
    resid_std = df['BSTS_Residuals'].std()
    df['Shock_Zone'] = np.where(np.abs(df['BSTS_Residuals']) > 1.5 * resid_std, 1, 0)
    
    df['is_forecast'] = 0
    df['Tension_Forecast'] = np.nan
    # Connect historical to future for clean plotting
    df.loc[df.index[-1], 'Tension_Forecast'] = df.loc[df.index[-1], 'BSTS_Expected_Balance']

    # --- 12-MONTH FORECAST ---
    print("[Modeling] Forecasting 12 months into the future...")
    try:
        forecast = res.get_forecast(steps=12)
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        df_future = pd.DataFrame(index=future_dates)
        
        df_future['Tension_Forecast'] = forecast.predicted_mean.values
        df_future['Systemic_Tension_Score'] = np.nan # No actual tension yet
        df_future['BSTS_Expected_Balance'] = np.nan
        df_future['Shock_Zone'] = 0
        df_future['is_forecast'] = 1
        
        # Merge back
        df = pd.concat([df, df_future])
    except Exception as e:
        print(f"[Modeling] Forecast Warning: {e}")
        
    return df

# =====================================================================
# STEP 4: Visualization for Pitching (Planetary Aesthetic)
# =====================================================================

def plot_planetary_chart(df):
    print("[Plotting] Generating Planetary Signal Visualization...")
    fig = go.Figure()

    # Axis 2: Supply/Demand Divergence Area (Background fill)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['S_D_Distance'],
            name="S/D Divergence Tension",
            fill='tozeroy', mode='none',
            fillcolor='rgba(255, 0, 255, 0.15)', # Neon Magenta
            yaxis='y2'
        )
    )

    # Axis 3: Inventory Levels (Bar chart, semi-transparent)
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Inventory_Z'],
            name="Inventory Normalized",
            marker_color='rgba(0, 255, 204, 0.3)', # Neon Cyan
            yaxis='y3'
        )
    )

    # Axis 1: Oil Price (Prominent Neon Green line)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['oil_price'],
            name="Oil Price (USD)",
            mode='lines',
            line=dict(color='#00FF00', width=3, shape='spline'),
            yaxis='y'
        )
    )

    # Highlight Shock Zones (Anomalies where structural tension indicates impending shocks)
    shock_dates = df[df['Shock_Zone'] == 1].index
    for sd in shock_dates:
        fig.add_vrect(
            x0=sd - pd.DateOffset(days=15),
            x1=sd + pd.DateOffset(days=15),
            fillcolor="#FF003C", # Neon Crimson Red
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    
    # Dummy trace to add Shock Zone to Legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color="rgba(255, 0, 60, 0.8)"),
            name="💥 Shock Zone (Planetary Signal)"
        )
    )

    # Format layout for Multi-Axis Dark Mode
    fig.update_layout(
        title=dict(
            text="🌍 <b style='color:#00FFCC'>PLANETARY SIGNALS</b>: Structural Energy Imbalance",
            font=dict(color="#FFF", size=26, family="Courier New")
        ),
        paper_bgcolor="#0A0A10", plot_bgcolor="#0A0A10",
        font=dict(color="#A0A0B0", family="Courier New"),
        xaxis=dict(domain=[0, 0.9], showgrid=False, zeroline=False),
        yaxis=dict(
            title="🎯 Oil Price ($)",
            title_font=dict(color="#00FF00"), tickfont=dict(color="#00FF00"),
            showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False
        ),
        yaxis2=dict(
            title="⚡ Divergence (Z-Score)",
            title_font=dict(color="#FF00FF"), tickfont=dict(color="#FF00FF"),
            anchor="x", overlaying="y", side="right", showgrid=False
        ),
        yaxis3=dict(
            title="🧊 Inventory (Z-Score)",
            title_font=dict(color="#00FFCC"), tickfont=dict(color="#00FFCC"),
            anchor="free", overlaying="y", side="right", position=0.98, showgrid=False
        ),
        legend=dict(
            x=0.02, y=0.98, bgcolor="rgba(10, 10, 16, 0.9)",
            bordercolor="#333", borderwidth=1, font=dict(color="#EEE")
        ),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    print("[Plotting] Plot generated! Displaying in browser window...")
    fig.show()

# =====================================================================
# INTERPRETATION: The Planetary Signal
# =====================================================================
def print_interpretation():
    print("\n" + "="*80)
    print("🌍 THE PLANETARY SIGNAL: INTERPRETING BSTS RESIDUALS")
    print("="*80)
    print("In a Bayesian Structural Time Series, the 'expected' balance represents the")
    print("normal, cyclical absorption of energy in the economy.")
    print("\nThe *Residuals*—the unobserved components not explained by regular rhythms—")
    print("capture sudden, exogenous structural shifts.")
    print("\nA persistent or extreme residual signifies that planetary constraints")
    print("(e.g., physical supply shocks, acute climatic anomalies, geostrategic events)")
    print("are violently overriding the expected equilibrium.")
    print("\nInstead of standard macroeconomic noise, these deviations act as a")
    print("'Planetary Signal'—an early warning that physical reality is diverging from")
    print("market expectations, foreshadowing impending structural 'Price Shocks'")
    print("before they are fully factored into the market.")
    print("="*80 + "\n")

# =====================================================================
# Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    df_integrated = build_dataset()
    df_features = feature_engineering(df_integrated)
    df_modeled = probabilistic_modeling(df_features)
    print("\nSample of Final Dataset (Head):")
    print(df_modeled[['Systemic_Tension_Score', 'Rolling_Corr_Score_Price', 'Divergence_Signal', 'Shock_Zone']].tail())
    print_interpretation()
    plot_planetary_chart(df_modeled)
