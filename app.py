import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime

# Import our backend data pipeline
from planetary_signals import build_dataset, feature_engineering, probabilistic_modeling

st.set_page_config(
    page_title="Oil Shock Early Warning System",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
    /* Professional 'Planetary & Earth Engineering' Dark Theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0A0A10;
        color: #A0A0B0;
    }
    [data-testid="stSidebar"] {
        background-color: #11111A;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    
    /* Typography tweaks for Terminal/Engineering feel */
    h1, h2, h3 {
        color: #E0E0E0 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* KPI Metrics styling */
    div[data-testid="stMetricValue"] {
        color: #00FFCC; /* Neon cyan for main metric values */
        font-family: 'Courier New', Courier, monospace;
        font-size: 2.2rem;
    }
    
    /* Dynamic Signal Explanation styling */
    .signal-box {
        background-color: rgba(255, 0, 60, 0.15); /* Slightly transparent crimson */
        border-left: 5px solid #FF003C;           /* Solid crimson border */
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 2rem;
        font-family: 'Courier New', Courier, monospace;
        color: #FF7788;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .signal-box.safe {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00FF00;
        color: #88FF88;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# BILINGUAL DICTIONARY (i18n)
# =====================================================================
i18n = {
    'EN': {
        'title': "Oil Shock Early Warning System",
        'sidebar_title': "🌍 Energy Analytics Lab",
        'config': "📌 Configuration",
        'perspective': "Perspective",
        'war_room': "🎯 War-Room Simulator",
        'enable_sim': "Enable Simulator Mode",
        'sim_caption': "Drag sliders to inject physical shocks and see predictive impacts.",
        'supply_shock': "⚡ Simulate: Supply Shortage",
        'demand_shock': "🔥 Simulate: Demand Surge",
        'fetching': "Fetching and Modeling Energy Physics...",
        'date_slider': "Select Historical Timeline",
        'current_tension': "Current Crisis Index",
        'inv_buffer': "Inventory Buffer (Days)",
        'price_dev': "Price Deviation (Expected Gap)",
        'signal_critical': "🚨 CRITICAL WARNING: Severe structural shortage detected. The physical system is collapsing below market expectations. A price shock is imminent.",
        'signal_warning': "⚠️ WARNING: The foundational supply mechanics are tightening rapidly. System is entering the Danger Zone.",
        'signal_safe': "✅ STABLE: System operating within expected cyclical limits. Forecast indicates stable trajectory.",
        'chart1_title': "I. Supply vs. Demand Fundamentals",
        'chart1_caption': "When Demand (Magenta) rises above Supply (Cyan), a physical shortage is triggered, causing System Tension.",
        'demand_label': "Demand Volume",
        'supply_label': "Supply Volume",
        'volume_z': "Volume (Index)",
        'chart2_title': "II. Shock Alerts & 1-Year Forecast",
        'chart2_caption': "Red dots indicate predictive warnings *before* price jumps. Dotted line shows the 1-year future equilibrium trend.",
        'expected_hist': "Expected Trend (Historical)",
        'forecast_line': "🔮 1-Year Future Trend",
        'oil_price': "Market Oil Price",
        'shock_alert': "🚨 Shock Warning",
        'price_axis': "Oil Price ($)",
        'tension_axis': "Crisis Index / Trend",
        'biz_title': "III. Business Value & Strategic Impact",
        'biz_desc': """
- **✈️ Supply Chain Hedging (Airlines/Logistics):** If the 'Shock Alert' flashes, logistics companies can instantly buy oil futures to lock in low prices before the physical shortage hits the financial market, saving billions in operational costs.
- **🏛️ National Security (Strategic Reserves):** Using the 1-year 'Expected Trend' forecast, the Ministry of Energy can precisely time when to refill the Strategic Petroleum Reserves or when to inject subsidies into the retail fuel fund proactively.
- **📈 Algorithmic Trading (Hedge Funds):** Integrating our API (which detects physical constraint) into automated trading bots allows funds to 'long' oil commodities weeks before technical traders realize a shock is occurring.
- **🍞 Macro-economic Inflation Forecasting:** Since energy drives CPI, predicting an oil shock 6 months in advance equals predicting inflation 6 months in advance. This gives Central Banks a massive early-warning advantage for setting interest rates.
"""
    },
    'TH': {
        'title': "ระบบเตือนภัยวิกฤตราคาน้ำมัน (Oil Shock Early Warning)",
        'sidebar_title': "🌍 ห้องปฏิบัติการพลังงาน",
        'config': "📌 การตั้งค่าพื้นฐาน",
        'perspective': "มุมมองวิเคราะห์",
        'war_room': "🎯 จำลองสถานการณ์สงคราม (Simulator)",
        'enable_sim': "เปิดใช้งานโหมดจำลอง (Simulator)",
        'sim_caption': "ปรับสไลเดอร์เพื่อดูผลกระทบล่วงหน้า หากเกิดวิกฤตความต้องการสวิง",
        'supply_shock': "⚡ ทดสอบ: กำลังผลิตหายกระทันหัน",
        'demand_shock': "🔥 ทดสอบ: ความต้องการพุ่งก้าวกระโดด",
        'fetching': "กำลังเชื่อมต่อคลังข้อมูลและวิเคราะห์โมเดล...",
        'date_slider': "เลือกช่วงเวลาการวิเคราะห์",
        'current_tension': "ดัชนีความเสี่ยงวิกฤต (Crisis Index)",
        'inv_buffer': "คลังน้ำมันสำรอง (หน่วย: วัน)",
        'price_dev': "ช่องว่างราคาส่วนเกิน (Price Gap)",
        'signal_critical': "🚨 วิกฤตระดับแดง: ถังน้ำมันโลกกำลังแห้งขอดสวนทางกับตลาด ระบบตรวจพบความพังทลายของระดับโครงสร้าง... ระวังราคาน้ำมันจะกระโดดขึ้นอย่างรุนแรง!",
        'signal_warning': "⚠️ เฝ้าระวัง: กำลังผลิตน้ำมันจริงเริ่มตึงมืออย่างรวดเร็ว ระบบกำลังเข้าสู่โซนอันตรายและราคาจ่อพุ่ง",
        'signal_safe': "✅ ปกติเสถียรภาพ: โครงสร้างพลังงานในปัจจุบันทำรอบได้เป็นปกติ โมเดลพยากรณ์คาดการณ์ว่าราคายังจะทรงตัวไปอีกระยะ",
        'chart1_title': "I. พื้นฐานชี้วัดการขาดแคลน (Supply vs. Demand)",
        'chart1_caption': "เมื่อกราฟ 'ความต้องการ' (สีชมพู) วิ่งทะลุสูงกว่า 'กำลังผลิต' (สีฟ้า) หมายถึงการเข้าสู่โซนขาดแคลนที่จะผลักดันราคาน้ำมัน",
        'demand_label': "ความต้องการ (Demand)",
        'supply_label': "กำลังผลิต (Supply)",
        'volume_z': "ปริมาณชี้วัด (Index)",
        'chart2_title': "II. แจ้งเตือนความผิดปกติ และพยากรณ์อนาคต 1 ปี",
        'chart2_caption': "จุด 'สีแดง' คือจุดที่ระบบยิงสัญญาณเตือนวิกฤตล่วงหน้า 'ก่อน' ที่ราคาจะพุ่งตามมา... ส่วนเส้นประคือพยากรณ์ทิศทางอนาคต",
        'expected_hist': "ทิศทางที่ควรจะเป็นในอดีต (Trend)",
        'forecast_line': "🔮 พยากรณ์ทิศทาง 1 ปีล่วงหน้า",
        'oil_price': "ราคาน้ำมันจริว (Market Price)",
        'shock_alert': "🚨 ยิงสัญญาณเตือนวิกฤต",
        'price_axis': "ราคาน้ำมัน ($)",
        'tension_axis': "ดัชนีความเสี่ยง",
        'biz_title': "III. การต่อยอดเพื่อสร้างมูลค่าทางธุรกิจ (Business Value & Impact)",
        'biz_desc': """
- **✈️ บริหารห่วงโซ่อุปทาน (สายการบิน/โลจิสติกส์):** ทันทีที่ระบบมี 'สัญญาณเตือนสีแดง' บริษัทสามารถเข้าล็อกราคาน้ำมันล่วงหน้า (Hedging Futures) ได้ทันทีก่อนที่ตลาดจะปรับตัวขึ้น ช่วยประหยัดต้นทุนดำเนินการมหาศาล
- **🏛️ ความมั่นคงระดับชาติ (นโยบายพลังงาน):** ใช้เส้นพยากรณ์ล่วงหน้า 1 ปี ให้รัฐบาลรู้ว่าช่วงไหนควรตุนน้ำมันเข้าคลังสำรอง (SPR) หรือเตรียมงบอุดหนุนกองทุนน้ำมันเชื้อเพลิงล่วงหน้าแบบไม่ต้องวัวหายล้อมคอก
- **📈 กองทุนเก็งกำไรอัตโนมัติ (Algorithmic Trading):** ระบบเราจับฟิสิกส์ของจริงได้ไวกว่ากราฟเทคนิค สามารถเอา API ไปผูกบอทเทรดเพื่อซื้อน้ำมันล่วงหน้า (Long Position) ดักรอนักเทรดฝั่งการเงินได้
- **🍞 การทำนายเงินเฟ้อล่วงหน้า (Macro-economics):** น้ำมันคือต้นน้ำทุกอย่าง ถ้าระบบทายว่าน้ำมันจะแพงในอีก 6 เดือนข้างหน้า แบงก์ชาติก็จะรู้ว่า 'เงินเฟ้อ (CPI)' กำลังจะพุ่ง และสามารถปรับอัตราดอกเบี้ยดักทางได้แม่นยำขึ้น
"""
    }
}

# Language Toggle Location
LANG_SELECT = st.sidebar.radio("🌐 Language / ภาษา", ["TH", "EN"])
t = i18n[LANG_SELECT]

# =====================================================================
# DATA CACHING & PROCESSING
# =====================================================================
@st.cache_data
def load_base_data():
    return build_dataset()

@st.cache_data
def get_features(df_base, country):
    return feature_engineering(df_base, country=country)

@st.cache_data
def get_modeled(df_features):
    return probabilistic_modeling(df_features)

# =====================================================================
# SIDEBAR FILTERS & WAR ROOM SIMULATOR
# =====================================================================
st.sidebar.title(t['sidebar_title'])
st.sidebar.markdown("---")
st.sidebar.subheader(t['config'])

# Perspective Selector
selected_country = st.sidebar.selectbox(t['perspective'], ["Global", "Thailand"])

# WAR ROOM SIMULATOR
st.sidebar.markdown("---")
st.sidebar.subheader(t['war_room'])

enable_simulator = st.sidebar.toggle(t['enable_sim'], value=False)

if enable_simulator:
    st.sidebar.caption(t['sim_caption'])
    supply_shock = st.sidebar.slider(t['supply_shock'], -3.0, 3.0, 0.0, step=0.1)
    demand_shock = st.sidebar.slider(t['demand_shock'], -3.0, 3.0, 0.0, step=0.1)
else:
    supply_shock = 0.0
    demand_shock = 0.0

with st.spinner(t['fetching']):
    df_base = load_base_data()
    df_f = get_features(df_base, selected_country)
    
    # ⚡ SHOCK INJECTION LOGIC ⚡
    if supply_shock != 0.0 or demand_shock != 0.0:
        df_sim = df_f.copy()
        # Apply shock onto the last 3 months
        shock_idx = df_sim.index[-3:]
        df_sim.loc[shock_idx, 'Supply_Z'] += supply_shock
        df_sim.loc[shock_idx, 'Demand_Z'] += demand_shock
        df_sim['Systemic_Tension_Score'] = (df_sim['Supply_Z'] - df_sim['Demand_Z']) - df_sim['Delta_Inventory_Z']
        df_sim['S_D_Distance'] = np.abs(df_sim['Supply_Z'] - df_sim['Demand_Z'])
        df = probabilistic_modeling(df_sim)
    else:
        df = get_modeled(df_f)

# Split sets
df_hist = df[df['is_forecast'] == 0]
df_fcst = df[df['is_forecast'] == 1]

# Date Slider
st.sidebar.markdown("---")
min_dt = df_hist.index.min().to_pydatetime()
max_dt = df_hist.index.max().to_pydatetime()
default_start = max_dt - datetime.timedelta(days=365 * 10) 

date_range = st.sidebar.slider(
    t['date_slider'],
    min_value=min_dt, max_value=max_dt,
    value=(default_start, max_dt),
    format="YYYY-MM"
)

# Apply Filter visually
df_filtered = df_hist[(df_hist.index >= date_range[0]) & (df_hist.index <= date_range[1])]

latest_record = df_filtered.iloc[-1]
latest_tension = latest_record['Systemic_Tension_Score']
latest_inv = latest_record['eia_ending_stocks']
latest_price = latest_record['oil_price']
buffer_days = max(10, min(120, (latest_inv / 1e6) * 30))
price_deviation = latest_tension - latest_record['BSTS_Expected_Balance']

# =====================================================================
# MAIN DASHBOARD UI
# =====================================================================
st.title(t['title'])

# 1. CLEAN EXECUTIVE KPIs
col1, col2, col3 = st.columns(3)

with col1:
    # Determine color dot for executive read
    if latest_tension < -1.5:
        status_dot = "🔴"
    elif latest_tension > 1.5:
        status_dot = "🟢"
    else:
        status_dot = "🟡"
        
    st.metric(
        label=f"{status_dot} {t['current_tension']}", 
        value=f"{latest_tension:.2f}",
        delta=f"{(latest_tension - df_filtered.iloc[-2]['Systemic_Tension_Score']):.2f} Index MoM",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label=t['inv_buffer'], 
        value=f"{buffer_days:.1f}",
        delta=f"{(latest_record['Delta_Inventory_Z']):.2f} Z-Score MoM",
        delta_color="normal"
    )

with col3:
    st.metric(
        label=t['price_dev'], 
        value=f"${latest_price:.2f}",
        delta=f"{-1 * price_deviation:.2f} Dev Gap",
        delta_color="off"
    )

# 2. SIGNAL EXPLANATION BOX
if latest_record['Shock_Zone'] == 1:
    signal_class = "signal-box"
    signal_msg = t['signal_critical']
elif latest_tension < -1.5:
    signal_class = "signal-box"
    signal_msg = t['signal_warning']
else:
    signal_class = "signal-box safe"
    signal_msg = t['signal_safe']

st.markdown(f'<div class="{signal_class}">{signal_msg}</div>', unsafe_allow_html=True)


# 3. MAIN CHARTS
st.subheader(t['chart1_title'])
st.caption(t['chart1_caption'])

df_chart = df_filtered.copy()
df_chart['Supply_Smooth'] = df_chart['Supply_Z'].rolling(3, min_periods=1).mean()
df_chart['Demand_Smooth'] = df_chart['Demand_Z'].rolling(3, min_periods=1).mean()

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart['Demand_Smooth'],
    name=t['demand_label'],
    mode='lines',
    line=dict(color='#FF00FF', width=3) # Magenta
))

fig1.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart['Supply_Smooth'],
    name=t['supply_label'],
    mode='lines',
    line=dict(color='#00FFCC', width=3), # Cyan
    fill='tonexty',
    fillcolor='rgba(255, 0, 60, 0.2)'
))

fig1.update_layout(
    height=320,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#A0A0B0', family="Courier New"),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(title=t['volume_z'], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    hovermode='x unified',
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

st.subheader(t['chart2_title'])
st.caption(t['chart2_caption'])

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart['BSTS_Expected_Balance'],
    name=t['expected_hist'],
    mode='lines',
    line=dict(color='rgba(255,255,255,0.3)', width=2)
))

fig2.add_trace(go.Scatter(
    x=df_fcst.index, y=df_fcst['Tension_Forecast'],
    name=t['forecast_line'],
    mode='lines',
    line=dict(color='#FF00FF', width=3, dash='dot')
))

fig2.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart['oil_price'],
    name=t['oil_price'],
    mode='lines',
    line=dict(color='#00FF00', width=3) # Neon Green
))

fig2.update_layout(
    yaxis=dict(title=t['tension_axis']),
    yaxis2=dict(title=t['price_axis'], title_font=dict(color="#00FF00"), tickfont=dict(color="#00FF00"), overlaying="y", side="right")
)

fig2.data[-1].update(yaxis='y2')

shocks = df_chart[df_chart['Shock_Zone'] == 1]
fig2.add_trace(go.Scatter(
    x=shocks.index, y=shocks['oil_price'],
    name=t['shock_alert'],
    mode='markers',
    marker=dict(color='#FF003C', size=12, symbol='circle', line=dict(color='#FFF', width=1)),
    yaxis='y2'
))

fig2.update_layout(
    height=350,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#A0A0B0', family="Courier New"),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    hovermode='x unified',
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig2, use_container_width=True)

# 4. BUSINESS VALUE & IMPACT
st.divider()
st.subheader(t['biz_title'])
st.markdown(t['biz_desc'], unsafe_allow_html=True)
