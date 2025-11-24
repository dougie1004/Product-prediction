import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. 페이지 설정 (Wide Mode & CSS Hack for Clean Look)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Executive Production Dashboard", layout="wide")

# CSS로 여백 줄이고 깔끔하게 만들기
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {font-size: 1.8rem !important;}
    h3 {font-size: 1.2rem !important; margin-bottom: 0px;}
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링 (Back-end)
# -----------------------------------------------------------------------------
# (1) 데이터 준비
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

# 데이터 평균값 (기준점용)
means = df_clean.mean()

# -----------------------------------------------------------------------------
# 3. 사이드바 (Input)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Simulation Control")
    st.markdown("---")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour)", 160, 200, 180, step=1)
    
    st.markdown("---")
    st.caption(f"Model Accuracy ($R^2$): **{model.rsquared:.2f}**")
    st.caption("Based on 28 months data")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 (Dashboard UI)
# -----------------------------------------------------------------------------
st.title("🏭 생산 실적 예측 대시보드")
st.markdown("AI Model Prediction based on Operational Inputs")

# (1) 예측 계산
input_data = pd.DataFrame({'const': 1.0, 'yield': [input_yield], 'productivity': [input_prod], 'workforce': [input_wf], 'hour': [input_hour]})
predictions = model.get_prediction(input_data)
pred_df = predictions.summary_frame(alpha=0.05)
pred_val = pred_df['mean'][0]
lower_val, upper_val = pred_df['obs_ci_lower'][0], pred_df['obs_ci_upper'][0]

# --- SECTION 1: 핵심 KPI (Top Row) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("예측 생산량 (Target)", f"{pred_val:.1f} 톤", delta=f"{pred_val - means['production']:.1f} vs Avg")
with col2:
    st.metric("최소 보장 (Risk Min)", f"{lower_val:.1f} 톤", delta="- Conservative", delta_color="off")
with col3:
    st.metric("최대 가능 (Max)", f"{upper_val:.1f} 톤", delta="+ Optimistic", delta_color="off")
with col4:
    # 달성률 (가상의 목표 100톤 대비)
    achievement = (pred_val / 100) * 100
    st.metric("목표 달성률 (Ref. 100t)", f"{achievement:.1f}%")

st.markdown("---")

# --- SECTION 2: 메인 게이지 차트 & 시계열 (Middle Row) ---
c_left, c_right = st.columns([1, 2])

with c_left:
    st.subheader("🎯 예측 계기판 (Gauge)")
    # Plotly Gauge Chart (자동차 속도계 스타일)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Production", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [lower_val*0.8, upper_val*1.1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ecc71"}, # 초록색 바
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [lower_val*0.8, lower_val], 'color': '#ffcdd2'}, # 위험구간 색상
                {'range': [lower_val, upper_val], 'color': '#f1f8e9'} # 안전구간 색상
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': pred_val
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c_right:
    st.subheader("📊 예측 범위 시각화")
    # Plotly Bar Chart with Error Bars (깔끔한 가로형 바)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['생산량'], x=[pred_val],
        orientation='h',
        marker_color='#2980b9',
        error_x=dict(type='data', array=[upper_val-pred_val], arrayminus=[pred_val-lower_val], color='red', width=5),
        text=[f"{pred_val:.1f}"], textposition='auto',
        hoverinfo='x+y'
    ))
    fig_bar.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Production (Tons)", range=[lower_val*0.8, upper_val*1.1]),
        plot_bgcolor='rgba(0,0,0,0)', # 투명 배경
        yaxis=dict(showticklabels=False) # Y축 라벨 숨김 (깔끔하게)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 3: 투입 변수 진단 (Bottom Row - Bullet Charts) ---
st.subheader("🔍 투입 변수 적정성 진단 (vs 과거 평균)")
st.caption("파란색 막대(현재 입력)가 회색 막대(과거 평균)보다 길면, 평균보다 높게 설정된 것입니다.")

# 4개의 컬럼에 작은 불렛 차트 배치
cols = st.columns(4)
vars_config = [
    ('yield', '수율 (%)', input_yield, means['yield'], 100),
    ('productivity', '생산성', input_prod, means['productivity'], 2.5),
    ('workforce', '인원 (명)', input_wf, means['workforce'], 70),
    ('hour', '작업시간 (h)', input_hour, means['hour'], 220)
]

for i, (col_name, title, curr, avg, max_range) in enumerate(vars_config):
    with cols[i]:
        # Bullet Chart 스타일
        fig_bullet = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = curr,
            domain = {'x': [0.1, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 14}},
            number = {'font': {'size': 20}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, max_range]},
                'bar': {'color': "#34495e"}, # 현재값 (진한 색)
                'bgcolor': "white",
                'steps': [
                    {'range': [0, avg], 'color': "#ecf0f1"} # 평균까지 (연한 회색)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': avg # 평균 위치에 빨간 선 표시
                }
            }
        ))
        fig_bullet.update_layout(height=120, margin=dict(l=15, r=15, t=10, b=10))
        st.plotly_chart(fig_bullet, use_container_width=True)
