import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 및 디자인 설정 (제목 잘림 해결 & 깔끔한 여백)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Executive Production Dashboard", layout="wide")

st.markdown("""
<style>
    /* 제목이 잘리지 않도록 상단 여백 확보 (1rem -> 3rem) */
    .block-container {
        padding-top: 3rem !important; 
        padding-bottom: 2rem !important;
    }
    /* KPI 박스 디자인 */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* 헤더 폰트 조정 */
    h1 { margin-top: 0px; }
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

means = df_clean.mean()

# -----------------------------------------------------------------------------
# 3. 사이드바 (Input Control)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Simulation Control")
    st.info("Change inputs to simulate outcomes.")
    st.markdown("---")
    
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Productivity)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (Workforce)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (Hour)", 160, 200, 180, step=1)
    
    st.markdown("---")
    st.caption(f"Model Accuracy ($R^2$): **{model.rsquared:.2f}**")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 (Dashboard UI)
# -----------------------------------------------------------------------------
st.title("🏭 생산 실적 예측 대시보드")
st.markdown("AI-driven Production Forecasting & Risk Analysis")
st.markdown("---")

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
    achievement = (pred_val / 100) * 100
    st.metric("목표 달성률 (Ref. 100t)", f"{achievement:.1f}%")

st.write("") # 간격 추가

# --- SECTION 2: 메인 차트 (Middle Row) ---
c_left, c_right = st.columns([1, 2])

with c_left:
    st.subheader("🎯 예측 계기판")
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': " 톤", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [lower_val*0.8, upper_val*1.1], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'steps': [
                {'range': [lower_val*0.8, lower_val], 'color': '#ffcdd2'},
                {'range': [lower_val, upper_val], 'color': '#f1f8e9'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_val}
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c_right:
    st.subheader("📊 예측 범위 상세")
    # Bar Chart (Tooltip Fixed)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['생산량'], x=[pred_val],
        orientation='h',
        marker_color='#2980b9',
        error_x=dict(type='data', array=[upper_val-pred_val], arrayminus=[pred_val-lower_val], color='red', width=5),
        text=[f"{pred_val:.1f} 톤"], 
        textposition='auto',
        # [중요] 툴팁 포맷 설정: 소수점 1자리까지만 표시
        hovertemplate='<b>예측값:</b> %{x:.1f} 톤<br>' +
                      '<b>안전 범위:</b> ±' + f"{(upper_val-lower_val)/2:.1f} 톤" + 
                      '<extra></extra>' 
    ))
    fig_bar.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Production (Tons)", range=[lower_val*0.8, upper_val*1.1]),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showticklabels=False),
        hoverlabel=dict(bgcolor="white", font_size=14) # 툴팁 디자인
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 3: 투입 변수 진단 (Bottom Row) ---
st.subheader("🔍 투입 변수 적정성 진단")
st.caption("현재 계획(진한 막대) vs 과거 평균(연한 회색)")

cols = st.columns(4)
vars_config = [
    ('yield', '수율 (%)', input_yield, means['yield'], 100),
    ('productivity', '생산성', input_prod, means['productivity'], 2.5),
    ('workforce', '인원 (명)', input_wf, means['workforce'], 70),
    ('hour', '작업시간 (h)', input_hour, means['hour'], 220)
]

for i, (col_name, title, curr, avg, max_range) in enumerate(vars_config):
    with cols[i]:
        fig_bullet = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = curr,
            domain = {'x': [0.1, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 14}},
            number = {'font': {'size': 18}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, max_range]},
                'bar': {'color': "#34495e"},
                'bgcolor': "white",
                'steps': [{'range': [0, avg], 'color': "#ecf0f1"}],
                'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': avg}
            }
        ))
        # 툴팁 간단하게 설정
        fig_bullet.update_traces(hovertemplate=f"<b>{title}</b><br>입력값: %{{x}}<br>평균: {avg:.1f}<extra></extra>")
        fig_bullet.update_layout(height=120, margin=dict(l=15, r=15, t=10, b=10))
        st.plotly_chart(fig_bullet, use_container_width=True)
