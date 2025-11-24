import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 화면 구성 설정 (CSS Hack for Compact Layout)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Production Dashboard", layout="wide")

# CSS: 여백 최소화 및 폰트 사이즈 조절
st.markdown("""
<style>
    /* 상단 여백 대폭 축소 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* 제목 및 헤더 여백 축소 */
    h1 { margin-bottom: 0px !important; font-size: 1.5rem !important; }
    h3 { margin-top: 10px !important; margin-bottom: 5px !important; font-size: 1.1rem !important; }
    
    /* KPI 메트릭 카드 디자인 및 여백 축소 */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 5px 10px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    
    /* 그래프 간격 조절 */
    .js-plotly-plot { margin-bottom: 0px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 및 모델링
# -----------------------------------------------------------------------------
np.random.seed(123)
n = 30
df = pd.DataFrame({
    'production': np.random.normal(100, 10, n),
    'yield': np.random.uniform(80, 95, n),
    'productivity': np.random.uniform(1.0, 2.0, n),
    'workforce': np.random.choice(range(40, 61), n),
    'hour': np.random.choice(range(160, 201), n)
})

drop_indices = [16, 19, 22]
df_clean = df.drop(drop_indices, errors='ignore').reset_index(drop=True)

X = df_clean[['yield', 'productivity', 'workforce', 'hour']]
y = df_clean['production']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
means = df_clean.mean()

# -----------------------------------------------------------------------------
# 3. 사이드바 (Inputs)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    input_yield = st.slider("수율 (Yield, %)", 80.0, 95.0, 88.0, step=0.1)
    input_prod = st.slider("생산성 (Index)", 1.0, 2.0, 1.5, step=0.1)
    input_wf = st.slider("투입 인원 (명)", 40, 60, 50, step=1)
    input_hour = st.slider("작업 시간 (h)", 160, 200, 180, step=1)
    st.divider()
    st.caption(f"Model Accuracy ($R^2$): {model.rsquared:.2f}")

# -----------------------------------------------------------------------------
# 4. 메인 대시보드 (Layout)
# -----------------------------------------------------------------------------
# (1) 헤더 및 KPI 영역
st.title("🏭 생산 실적 예측 대시보드")

# 예측 계산
input_data = pd.DataFrame({'const': 1.0, 'yield': [input_yield], 'productivity': [input_prod], 'workforce': [input_wf], 'hour': [input_hour]})
predictions = model.get_prediction(input_data)
pred_df = predictions.summary_frame(alpha=0.05)
pred_val = pred_df['mean'][0]
lower_val, upper_val = pred_df['obs_ci_lower'][0], pred_df['obs_ci_upper'][0]

# KPI 배치 (Top Row)
k1, k2, k3, k4 = st.columns(4)
k1.metric("예측 생산량 (Target)", f"{pred_val:.1f} 톤", delta=f"{pred_val - means['production']:.1f}")
k2.metric("최소 보장 (Risk Min)", f"{lower_val:.1f} 톤", delta_color="off")
k3.metric("최대 가능 (Max)", f"{upper_val:.1f} 톤", delta_color="off")
k4.metric("목표 달성률 (Ref. 100t)", f"{(pred_val/100)*100:.1f}%")

# (2) 메인 차트 영역 (Middle Row)
# 높이를 220px로 줄여서 한 화면에 들어오게 함
c_left, c_right = st.columns([1, 2]) # 비율 1:2

with c_left:
    st.subheader("🎯 예측 계기판")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        number = {'font': {'size': 24}}, # 글자 크기 최적화
        gauge = {
            'axis': {'range': [lower_val*0.8, upper_val*1.1]},
            'bar': {'color': "#2ecc71"},
            'steps': [
                {'range': [lower_val*0.8, lower_val], 'color': '#ffcdd2'},
                {'range': [lower_val, upper_val], 'color': '#f1f8e9'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_val}
        }
    ))
    # 마진 제거 및 높이 축소
    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10)) 
    st.plotly_chart(fig_gauge, use_container_width=True)

with c_right:
    st.subheader("📊 예측 범위 상세")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=['생산량'], x=[pred_val],
        orientation='h',
        marker_color='#2980b9',
        error_x=dict(type='data', array=[upper_val-pred_val], arrayminus=[pred_val-lower_val], color='red', width=3),
        text=[f"{pred_val:.1f}"], textposition='auto'
    ))
    fig_bar.update_layout(
        height=200, # 높이 축소
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[lower_val*0.8, upper_val*1.1]),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# (3) 하단 입력 변수 진단 (Bottom Row)
st.subheader("🔍 변수 적정성 진단 (vs 과거 평균)")

cols = st.columns(4)
vars_config = [
    ('yield', '수율', input_yield, means['yield'], 100),
    ('productivity', '생산성', input_prod, means['productivity'], 2.5),
    ('workforce', '인원', input_wf, means['workforce'], 70),
    ('hour', '시간', input_hour, means['hour'], 220)
]

for i, (col_name, title, curr, avg, max_val) in enumerate(vars_config):
    with cols[i]:
        fig_bullet = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = curr,
            title = {'text': title, 'font': {'size': 12}}, # 폰트 작게
            number = {'font': {'size': 18}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, max_val]},
                'bar': {'color': "#34495e"},
                'steps': [{'range': [0, avg], 'color': "#ecf0f1"}],
                'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': avg}
            }
        ))
        # 불필요한 마진 제거 및 초소형 높이 설정
        fig_bullet.update_layout(height=80, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bullet, use_container_width=True)
