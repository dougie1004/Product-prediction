import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 및 디자인 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Executive Production Dashboard", layout="wide")

st.markdown("""
<style>
    /* 1. 제목 잘림 방지 (상단 여백 확보) */
    .block-container {
        padding-top: 3rem !important; 
        padding-bottom: 2rem !important;
    }
    /* 2. KPI 박스 입체감 주기 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* 3. 전체 배경색 미세 조정 (옵션) */
    .stApp {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링 (Back-end)
# -----------------------------------------------------------------------------
# (1) 데이터 준비 (데모 데이터)
np.random.seed(123)
n = 30
df = pd.DataFrame({
    'production': np.random.normal(100, 10, n),
    'yield': np.random.uniform(80, 95, n),
    'productivity': np.random.uniform(1.0, 2.0, n),
    'workforce': np.random.choice(range(40, 61), n),
    'hour': np.random.choice(range(160, 201), n)
})

# (2) 전처리 & 모델링
drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 과거 데이터 평균값 (비교 기준용)
means = df_clean.mean()

# -----------------------------------------------------------------------------
# 3. 사이드바 (Input Control)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 설정")
    st.info("조건을 변경하면 우측 대시보드에 실시간으로 반영됩니다.")
    st.markdown("---")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour)", 160, 200, 180, step=1)
    
    st.markdown("---")
    st.caption(f"Model Accuracy ($R^2$): **{model.rsquared:.2f}**")
    st.caption("Data Source: 2020.01 ~ 2022.04")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 (Dashboard UI)
# -----------------------------------------------------------------------------
st.title("🏭 생산 실적 예측 대시보드")
st.markdown("**AI-driven Production Forecasting & Risk Analysis**")
st.write("") # 여백 추가

# (1) 예측 계산 Logic
input_data = pd.DataFrame({'const': 1.0, 'yield': [input_yield], 'productivity': [input_prod], 'workforce': [input_wf], 'hour': [input_hour]})
predictions = model.get_prediction(input_data)
pred_df = predictions.summary_frame(alpha=0.05)

pred_val = pred_df['mean'][0]
lower_val = pred_df['obs_ci_lower'][0]
upper_val = pred_df['obs_ci_upper'][0]

# --- SECTION 1: 핵심 KPI (Top Row) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("예측 생산량 (Target)", f"{pred_val:.1f} 톤", delta=f"{pred_val - means['production']:.1f} vs Avg")
with col2:
    st.metric("최소 보장 (Risk Min)", f"{lower_val:.1f} 톤", delta="- Conservative", delta_color="off")
with col3:
    st.metric("최대 가능 (Max)", f"{upper_val:.1f} 톤", delta="+ Optimistic", delta_color="off")
with col4:
    achievement = (pred_val / 100) * 100
    st.metric("목표 달성률 (Ref. 100t)", f"{achievement:.1f}%")

st.markdown("---")

# --- SECTION 2: 메인 차트 (Middle Row) ---
c_left, c_right = st.columns([1, 2])

with c_left:
    st.subheader("🎯 예측 계기판")
    # Gauge Chart (속도계 스타일)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': " 톤", 'font': {'size': 24, 'color': '#2c3e50'}},
        gauge = {
            'axis': {'range': [lower_val*0.8, upper_val*1.1], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"}, # 초록색 바
            'bgcolor': "white",
            'steps': [
                {'range': [lower_val*0.8, lower_val], 'color': '#ffcdd2'}, # 붉은색(위험)
                {'range': [lower_val, upper_val], 'color': '#f1f8e9'}      # 연두색(안전)
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness':
